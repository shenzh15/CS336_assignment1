"""
Training script for transformer language model.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from cs336_basics.model import transformer_lm
from cs336_basics.optimizer import AdamW, SGD, get_lr_cosine_schedule, clip_grad_norm
from cs336_basics.loss import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.serialization import save_checkpoint, load_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging only to console.")


class TrainingConfig:
    """Configuration class for training hyperparameters."""
    
    def __init__(self):
        # Model hyperparameters
        self.vocab_size = 32000
        self.context_length = 1024
        self.d_model = 768
        self.num_layers = 12
        self.num_heads = 12
        self.d_ff = None  # Will be computed as 8/3 * d_model if None
        self.rope_theta = 10000.0
        
        # Training hyperparameters
        self.batch_size = 16                    # Number of samples per training batch
        self.learning_rate = 1e-4               # Initial learning rate for optimizer
        self.min_learning_rate = 1e-5           # Minimum learning rate for cosine schedule
        self.weight_decay = 0.1                 # L2 regularization strength
        self.warmup_iters = 1000                # Number of iterations for learning rate warmup
        self.max_iters = 10000                  # Total number of training iterations
        self.eval_interval = 500                # Evaluate on validation set every N iterations
        self.eval_iters = 100                   # Number of batches to use for evaluation
        self.log_interval = 100                 # Log training metrics every N iterations
        self.save_interval = 1000               # Save checkpoint every N iterations
        
        # Optimizer settings
        self.optimizer_type = "adamw"  # "adamw" or "sgd"
        self.betas = (0.9, 0.95)
        self.eps = 1e-8
        
        # Data settings
        self.train_data_path = "runspace/tokenized_data/TinyStoriesV2-GPT4-train_tokens.npy"
        self.val_data_path = "runspace/tokenized_data/TinyStoriesV2-GPT4-valid_tokens.npy"
        
        # Checkpoint and logging
        self.checkpoint_dir = "checkpoints"
        self.resume_from = None
        self.use_wandb = False
        self.wandb_project = "cs336-transformer-training"
        self.wandb_run_name = None
        
        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.compile_model = False  # Use torch.compile for speedup
        
        # Gradient clipping
        self.grad_clip = 3.0  # Maximum L2 norm for gradient clipping (set to None to disable)
        
    def from_dict(self, config_dict: Dict[str, Any]):
        """Update config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""        
        config_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                # Handle non-serializable types
                if isinstance(v, torch.dtype):
                    config_dict[k] = str(v)
                else:
                    config_dict[k] = v
        return config_dict


class MemoryEfficientDataLoader:
    """Memory-efficient data loader using np.memmap."""
    
    def __init__(self, data_path: str, batch_size: int, context_length: int, device: str, vocab_size: int):
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.vocab_size = vocab_size
        
        # Load data using memory mapping
        self.data = np.load(data_path, mmap_mode='r')
        self.data_size = len(self.data)
        
        print(f"Loaded {data_path} with {self.data_size:,} tokens")
        
        if self.data_size < context_length + 1:
            raise ValueError(
                f"Dataset too small. Need at least {context_length + 1} tokens, "
                f"but got {self.data_size}"
            )
    
    def get_batch(self):
        """Get a batch of data using the existing get_batch function."""
        X, Y = get_batch(self.data, self.batch_size, self.context_length, self.device)
        
        # Verify that the memory-mapped data looks correct
        # Check for values beyond expected vocabulary size
        max_token_x = torch.max(X).item()
        max_token_y = torch.max(Y).item()
        min_token_x = torch.min(X).item()
        min_token_y = torch.min(Y).item()
        
        if min_token_x < 0 or min_token_y < 0:
            raise ValueError(f"Found negative token IDs: X_min={min_token_x}, Y_min={min_token_y}")
        
        if max_token_x >= self.vocab_size or max_token_y >= self.vocab_size:
            raise ValueError(
                f"Found token IDs beyond vocabulary size {self.vocab_size}: "
                f"X_max={max_token_x}, Y_max={max_token_y}"
            )
        
        return X, Y


class ModelMonitor:
    """Monitor activation norms, weight norms, and gradient norms during training."""
    
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup forward hooks on key layers."""
        # Monitor key layers following standard practice
        target_modules = [
            self.model.token_embedding,
            self.model.ln,
            self.model.lm_head
        ]
        
        # Add first and last transformer layers
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            target_modules.extend([
                self.model.layers[0],   # First layer
                self.model.layers[-1]   # Last layer
            ])
        
        # Register hooks
        for module in target_modules:
            hook = module.register_forward_hook(self._activation_hook)
            self.hooks.append(hook)
    
    def _activation_hook(self, module, input, output):
        """Hook function to capture activations."""
        # Suppress unused parameter warnings - these are required by PyTorch hook signature
        _ = module, input
        
        if isinstance(output, torch.Tensor):
            self.activations.append(output.detach())
        elif isinstance(output, tuple) and len(output) > 0:
            # Handle cached models that return (output, cache)
            if isinstance(output[0], torch.Tensor):
                self.activations.append(output[0].detach())
    
    def get_norms(self):
        """Get all monitoring metrics: activation, weight, and gradient norms."""
        norms = {}
        
        # Activation norm
        if self.activations:
            activation_norm = torch.nn.utils.get_total_norm(self.activations, norm_type=2.0)
            norms["activation_norm"] = activation_norm.item()
        else:
            norms["activation_norm"] = 0.0
        
        # Weight and gradient norms
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        # Weight norm
        weight_norm = torch.nn.utils.get_total_norm(parameters, norm_type=2.0)
        norms["weight_norm"] = weight_norm.item()
        
        # Gradient norm
        gradients = [p.grad for p in parameters if p.grad is not None]
        if gradients:
            grad_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
            norms["grad_norm"] = grad_norm.item()
        else:
            norms["grad_norm"] = 0.0
        
        return norms
    
    def clear_activations(self):
        """Clear captured activations to prevent memory leak."""
        self.activations.clear()
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, use_wandb: bool = False, wandb_project: str = None, 
                 wandb_run_name: str = None, wandb_resume_id: str = None, 
                 config: TrainingConfig = None):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if self.use_wandb:
            # Prepare wandb.init parameters
            init_kwargs = {
                "project": wandb_project,
                "name": wandb_run_name,
                "config": config.to_dict() if config else None
            }
            
            # Add resume parameters if wandb_resume_id is provided
            if wandb_resume_id:
                init_kwargs["id"] = wandb_resume_id
                init_kwargs["resume"] = "must"  # Must resume from this exact run
                print(f"Resuming Weights & Biases run with ID: {wandb_resume_id}")
            
            wandb.init(**init_kwargs)
            print("Initialized Weights & Biases logging")
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to console and wandb."""
        # Console logging
        log_str = f"Step {step:6d}"
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f" | {key}: {value:.4f}"
            else:
                log_str += f" | {key}: {value}"
        print(log_str)
        
        # Wandb logging
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def finish(self):
        """Clean up logger."""
        if self.use_wandb:
            wandb.finish()


def estimate_loss(model, train_loader, val_loader, eval_iters: int, config: TrainingConfig):
    """Estimate training and validation loss."""
    model.eval()
    losses = {}
    
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            with torch.no_grad():
                X, Y = loader.get_batch()
                # Disable KV cache for evaluation to save memory
                logits = model(X, use_cache=False)
                # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
                logits = logits.view(-1, logits.size(-1))
                targets = Y.view(-1)
                loss = cross_entropy(logits, targets)
                total_loss += loss.item()
        
        losses[split] = total_loss / eval_iters
    
    model.train()
    return losses


def create_model(config: TrainingConfig) -> transformer_lm:
    """Create and initialize the transformer model."""
    model = transformer_lm(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
        dtype=config.dtype
    )
    
    model = model.to(config.device)
    
    # Compile model for potential speedup (PyTorch 2.0+)
    if config.compile_model:
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Failed to compile model: {e}")
    
    return model


def create_optimizer(model: torch.nn.Module, config: TrainingConfig):
    """Create optimizer based on config."""
    if config.optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=config.learning_rate
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
    
    return optimizer


def get_learning_rate(iteration: int, config: TrainingConfig) -> float:
    """Get learning rate for current iteration using cosine schedule."""
    return get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_iters=config.warmup_iters,
        cosine_cycle_iters=config.max_iters
    )


def train_model(config: TrainingConfig, wandb_resume_id: str = None):
    """Main training function."""
    print("Starting training with configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader = MemoryEfficientDataLoader(
        config.train_data_path, config.batch_size, config.context_length, config.device, config.vocab_size
    )
    val_loader = MemoryEfficientDataLoader(
        config.val_data_path, config.batch_size, config.context_length, config.device, config.vocab_size
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Using {config.optimizer_type} optimizer")
    
    # Initialize logger
    logger = TrainingLogger(
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        config=config
    )
    
    # Initialize model monitor
    model_monitor = ModelMonitor(model)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if config.resume_from:
        print(f"Resuming from checkpoint: {config.resume_from}")
        start_iteration = load_checkpoint(config.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    print("Starting training loop...")
    model.train()
    
    # Initial evaluation
    if start_iteration == 0 :
        print("Running initial evaluation...")
        losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config)
        logger.log({
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "learning_rate": get_learning_rate(0, config)
        }, 0)
    
    training_start_time = time.time()
    
    for iteration in range(start_iteration + 1, config.max_iters + 1):
        # Update learning rate
        lr = get_learning_rate(iteration, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch and forward pass
        X, Y = train_loader.get_batch()
        
        # Forward pass (disable KV cache for training to save memory)
        logits = model(X, use_cache=False)
        
        # Compute loss
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        targets = Y.view(-1)
        loss = cross_entropy(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        

        # Gradient clipping using SAME parameter list
        if config.grad_clip is not None:
            clip_grad_norm(list(model.parameters()), config.grad_clip)  
            # there is a sad story, see: https://shenzh15.github.io/AI_Learning_note/#/gradient_clipping_bug_story

        optimizer.step()
        
        # Logging
        if iteration % config.log_interval == 0:
            # Get monitoring metrics (activations from current batch only)
            norms = model_monitor.get_norms()
            
            # Log training metrics including norms
            metrics = {
                "train_loss": loss.item(),
                "learning_rate": lr,
                "iteration": iteration,
                "activation_norm": norms["activation_norm"],
                "weight_norm": norms["weight_norm"],
                "grad_norm": norms["grad_norm"]
            }
            logger.log(metrics, iteration)
        
        # Always clear activations after each iteration to prevent accumulation
        model_monitor.clear_activations()
        
        # Evaluation
        if iteration % config.eval_interval == 0:
            print(f"Running evaluation at iteration {iteration}...")
            losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config)
            
            elapsed_time = time.time() - training_start_time
            
            logger.log({
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "learning_rate": lr,
                "elapsed_time": elapsed_time,
                "tokens_processed": iteration * config.batch_size * config.context_length
            }, iteration)
        
        # Checkpointing
        if iteration % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt"
            )
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_checkpoint.pt")
    print(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, config.max_iters, final_checkpoint_path)
    
    # Final evaluation
    print("Running final evaluation...")
    losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config)
    total_time = time.time() - training_start_time
    
    logger.log({
        "final_train_loss": losses["train"],
        "final_val_loss": losses["val"],
        "total_training_time": total_time,
        "final_tokens_processed": config.max_iters * config.batch_size * config.context_length
    }, config.max_iters)
    
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final train loss: {losses['train']:.4f}")
    print(f"Final validation loss: {losses['val']:.4f}")
    
    # Cleanup
    model_monitor.cleanup()
    logger.finish()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train transformer language model")
    
    # Configuration file
    parser.add_argument(
        "--config", type=str, help="Path to JSON config file"
    )
    
    # Model hyperparameters
    parser.add_argument("--vocab-size", type=int, dest="vocab_size", help="Vocabulary size")
    parser.add_argument("--context-length", type=int, dest="context_length", help="Context length")
    parser.add_argument("--d-model", type=int, dest="d_model", help="Model dimension")
    parser.add_argument("--num-layers", type=int, dest="num_layers", help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, dest="num_heads", help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, dest="d_ff", help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, dest="rope_theta", help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", help="Learning rate")
    parser.add_argument("--min-learning-rate", type=float, dest="min_learning_rate", help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, dest="weight_decay", help="Weight decay")
    parser.add_argument("--warmup-iters", type=int, dest="warmup_iters", help="Warmup iterations")
    parser.add_argument("--max-iters", type=int, dest="max_iters", help="Maximum iterations")
    parser.add_argument("--eval-interval", type=int, dest="eval_interval", help="Evaluation interval")
    parser.add_argument("--eval-iters", type=int, dest="eval_iters", help="Evaluation iterations")
    parser.add_argument("--log-interval", type=int, dest="log_interval", help="Logging interval")
    parser.add_argument("--save-interval", type=int, dest="save_interval", help="Save interval")
    
    # Optimizer settings
    parser.add_argument("--optimizer-type", type=str, choices=["adamw", "sgd"], dest="optimizer_type",
                       help="Optimizer type")
    parser.add_argument("--beta1", type=float, help="Adam beta1 parameter")
    parser.add_argument("--beta2", type=float, help="Adam beta2 parameter")
    parser.add_argument("--eps", type=float, help="Adam epsilon parameter")
    
    # Data settings
    parser.add_argument("--train-data-path", type=str, dest="train_data_path", help="Training data path")
    parser.add_argument("--val-data-path", type=str, dest="val_data_path", help="Validation data path")
    
    # Checkpoint and logging
    parser.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--resume-from", type=str, dest="resume_from", help="Resume from checkpoint")
    parser.add_argument("--use-wandb", action="store_true", dest="use_wandb", help="Use Weights & Biases")
    parser.add_argument("--wandb-project", type=str, dest="wandb_project", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, dest="wandb_run_name", help="W&B run name")
    
    # Device settings
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--compile-model", action="store_true", dest="compile_model", help="Compile model")
    
    # Wandb resume (command-line only, not saved in config)
    parser.add_argument("--wandb-resume-id", type=str, dest="wandb_resume_id", help="Wandb run ID to resume from (command-line only)")
    
    args = parser.parse_args()
    
    # Create config with defaults
    config = TrainingConfig()
    
    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config.from_dict(config_dict)
        print(f"Loaded configuration from {args.config}")
    
    # Extract wandb_resume_id before processing other arguments (not saved in config)
    wandb_resume_id = args.wandb_resume_id
    if wandb_resume_id:
        print(f"Will resume Weights & Biases run with ID: {wandb_resume_id}")
    
    # Override with command line arguments
    # For boolean flags (store_true), we need special handling
    bool_flags = {'use_wandb', 'compile_model'}
    # Parameters that should not be saved in config (command-line only)
    excluded_from_config = {'config', 'wandb_resume_id'}
    
    for key, value in vars(args).items():
        if key in excluded_from_config:
            continue
            
        # For boolean flags, override if True (user specified the flag)
        if key in bool_flags:
            if value:  # Only override if True (flag was specified)
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"Overriding {key}: {getattr(config, key)}")
                else:
                    print(f"Warning: Unknown config key from args: {key}")
        # For other arguments, override if not None
        elif value is not None:
            if key in ['beta1', 'beta2']:
                # Handle beta values specially
                betas = list(config.betas)
                betas[0 if key == 'beta1' else 1] = value
                config.betas = tuple(betas)
                print(f"Overriding betas: {config.betas}")
            elif hasattr(config, key):
                setattr(config, key, value)
                print(f"Overriding {key}: {getattr(config, key)}")
            else:
                print(f"Warning: Unknown config key from args: {key}")
    
    # Set device if not specified
    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {config.device}")
    print(f"Gradient clipping threshold: {config.grad_clip}")
    
    # Start training
    train_model(config, wandb_resume_id=wandb_resume_id)


if __name__ == "__main__":
    main()
