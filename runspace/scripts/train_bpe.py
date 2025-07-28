import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cs336_basics.train_bpe import train_bpe


def train_and_save_bpe(
    input_file: str,
    vocab_size: int,
    output_name: str,
    special_tokens: list[str] = None,
    show_progress: bool = True,
    output_dir: str = None
):
    """
    Train BPE model and save vocabulary and merge rules
    
    Args:
        input_file: Input file path
        vocab_size: Vocabulary size
        output_name: Output file name prefix
        special_tokens: Special token list, default is ["<|endoftext|>"]
        show_progress: Whether to show progress bar
        output_dir: Output directory, default is current directory
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(__file__))
    
    print(f"Starting BPE training...")
    print(f"Input file: {input_file}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Train BPE model
    id_to_token, merged = train_bpe(
        input_file, 
        vocab_size, 
        special_tokens, 
        show_progress=show_progress
    )
    
    # Save vocabulary
    vocab_file = os.path.join(output_dir, f"tokenizer_vocab_{output_name}.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({
            str(k): v.hex() for k, v in id_to_token.items()
        }, f, indent=2)
    
    # Save merge rules
    merges_file = os.path.join(output_dir, f"tokenizer_merges_{output_name}.txt")
    with open(merges_file, "w", encoding="utf-8") as f:
        for a, b in merged:
            f.write(f"{a.hex()}\t{b.hex()}\n")
    
    print(f"Training completed!")
    print(f"Vocabulary saved to: {vocab_file}")
    print(f"Merge rules saved to: {merges_file}")
    
    return id_to_token, merged


def train_owt_bpe():
    """Train BPE model on OWT dataset"""
    return train_and_save_bpe(
        input_file="../data/owt_train.txt",
        vocab_size=32000,
        output_name="owt_32k",
        special_tokens=["<|endoftext|>"],
        show_progress=True
    )


def train_tinystories_bpe():
    """Train BPE model on TinyStories dataset"""
    return train_and_save_bpe(
        input_file="../data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        output_name="tinystories_10k",
        special_tokens=["<|endoftext|>"],
        show_progress=True
    )


if __name__ == "__main__":
    # Choose which dataset to train
    print("Select dataset to train:")
    print("1. OWT (32k vocabulary)")
    print("2. TinyStories (10k vocabulary)")
    print("3. Both (train both datasets)")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        train_owt_bpe()
    elif choice == "2":
        train_tinystories_bpe()
    elif choice == "3":
        print("Starting training for both datasets...")
        print("\n=== Training TinyStories dataset ===")
        train_tinystories_bpe()
        print("\n=== Training OWT dataset ===")
        train_owt_bpe()
        print("\n=== All training completed! ===")
    else:
        print("Invalid choice!")