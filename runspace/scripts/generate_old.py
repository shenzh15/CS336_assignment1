#!/usr/bin/env python3
"""
Text generation script - Generate text using a trained Transformer model

Usage:
    python generate.py --checkpoint checkpoints/model.pt --tokenizer_path tokenizer --prompt "Once upon a time"
    
Or load parameters from config file:
    python generate.py --config config.json --checkpoint checkpoints/model.pt --prompt "Hello world"
"""

import argparse
import json
import os
import sys
import torch

# Add project root directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cs336_basics.model_old import transformer_lm
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.serialization import load_checkpoint
from cs336_basics.optimizer import AdamW  # Required for checkpoint loading


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_model_from_config(config: dict, device: torch.device) -> transformer_lm:
    """Create model from configuration"""
    # Use default value if d_ff is null
    d_ff = config.get('d_ff')
    if d_ff is None:
        d_ff = 4 * config['d_model']  # Standard 4x expansion
        
    model = transformer_lm(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=d_ff,
        rope_theta=config['rope_theta'],
        device=device
    )
    return model


def create_model_from_args(args, device: torch.device) -> transformer_lm:
    """Create model from command line arguments"""
    d_ff = args.d_ff if args.d_ff is not None else 4 * args.d_model
    
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=device
    )
    return model


def load_model_from_checkpoint(checkpoint_path: str, model: transformer_lm) -> transformer_lm:
    """Load model weights from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Create dummy optimizer (required for checkpoint loading)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Load checkpoint
    iteration = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"Loaded checkpoint from iteration: {iteration}")
    
    return model


def load_tokenizer(tokenizer_path: str, vocab_filename: str = "vocab.json", 
                   merges_filename: str = "merges.txt", special_tokens: list = None) -> Tokenizer:
    """Load tokenizer"""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    vocab_path = os.path.join(tokenizer_path, vocab_filename)
    merges_path = os.path.join(tokenizer_path, merges_filename)
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Merges file not found: {merges_path}")
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    print(f"  Vocab file: {vocab_filename}")
    print(f"  Merges file: {merges_filename}")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.id_to_token)}")
    
    return tokenizer


def interactive_generate(model: transformer_lm, tokenizer: Tokenizer, args):
    """Interactive generation mode"""
    print("\n" + "="*50)
    print("Interactive Text Generation Mode")
    print("Enter 'quit' or 'exit' to quit")
    print("Enter 'config' to view current generation parameters")
    print("="*50)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Exiting generator...")
                break
            elif prompt.lower() == 'config':
                print(f"Current parameters:")
                print(f"  Max tokens: {args.max_tokens}")
                print(f"  Temperature: {args.temperature}")
                print(f"  Top-p: {args.top_p}")
                continue
            elif not prompt:
                continue
            
            print(f"\nGenerating...")
            generated_text = model.decode(
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            print("\nGenerated text:")
            print("-" * 40)
            print(f"{prompt}{generated_text}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nUser interrupted, exiting...")
            break
        except Exception as e:
            print(f"Error during generation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Text generation script based on Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
                 epilog="""
Example usage:
  # Generate from config file and checkpoint
  python generate.py --config config_examples/TinyStory.json --checkpoint checkpoints/tiny_story/final_checkpoint.pt --tokenizer-path ../tokenizer_models/TinyStory --prompt "Once upon a time"
  
  # Specify model parameters directly
  python generate.py --checkpoint checkpoints/model.pt --tokenizer-path tokenizer --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --prompt "Hello world"
  
  # Interactive mode
  python generate.py --config config.json --checkpoint model.pt --tokenizer-path tokenizer --interactive
  
  # Use custom tokenizer file names
  python generate.py --checkpoint model.pt --tokenizer-path tokenizer --vocab-filename my_vocab.json --merges-filename my_merges.txt --prompt "Hello"
  
  # Generate with custom parameters  
  python generate.py --config config.json --checkpoint model.pt \\
    --tokenizer-path tokenizer --prompt "Hello"
        """
    )
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--tokenizer-path", type=str, required=True, dest="tokenizer_path",
                        help="Path to tokenizer directory")
    parser.add_argument("--vocab-filename", type=str, default="vocab.json", dest="vocab_filename",
                        help="Name of vocabulary file (default: vocab.json)")
    parser.add_argument("--merges-filename", type=str, default="merges.txt", dest="merges_filename",
                        help="Name of merges file (default: merges.txt)")
    
    # Config file or direct parameters
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", type=str,
                       help="Path to config file (JSON format)")
    
    # Model parameters (required when not using config file)
    parser.add_argument("--vocab-size", type=int, default=10000, dest="vocab_size",
                        help="Size of vocabulary")
    parser.add_argument("--context-length", type=int, default=256, dest="context_length",
                        help="Context length")
    parser.add_argument("--d-model", type=int, default=512, dest="d_model",
                        help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, dest="num_layers",
                        help="Number of Transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, dest="num_heads",
                        help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=None, dest="d_ff",
                        help="Feed-forward dimension (default: 4*d_model)")
    parser.add_argument("--rope-theta", type=float, default=10000.0, dest="rope_theta",
                        help="RoPE theta parameter")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="",
                        help="Input prompt (empty for interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100, dest="max_tokens",
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p",
                        help="Top-p sampling threshold (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive generation mode")
    
    # Device selection
    parser.add_argument("--device", type=str, default="auto",
                        help="Device selection: auto, cpu, cuda")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed for reproducible generation
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(args.tokenizer_path, args.vocab_filename, args.merges_filename)
        
        # Create model
        if args.config:
            print(f"Loading model parameters from config file: {args.config}")
            config = load_config(args.config)
            model = create_model_from_config(config, device)
        else:
            print("Creating model from command line arguments")
            model = create_model_from_args(args, device)
        
        print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load checkpoint
        model = load_model_from_checkpoint(args.checkpoint, model)
        model.eval()  # Set to evaluation mode
        
        
        # Generate text
        if args.interactive or not args.prompt.strip():
            interactive_generate(model, tokenizer, args)
        else:
            print("\nGenerating...")
            print(f"Prompt: {args.prompt}")
            
            generated_text = model.decode(
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            print("\nGenerated text:")
            print("="*60)
            print(f"{args.prompt}{generated_text}")
            print("="*60)
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
