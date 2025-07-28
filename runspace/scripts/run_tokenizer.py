#!/usr/bin/env python3
"""
Large file tokenizer encoder
This program processes large files in the data directory, encodes them using the appropriate tokenizer,
and saves the results as uint16 NumPy arrays.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Iterator, List
import argparse
from tqdm import tqdm

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cs336_basics.tokenizer import Tokenizer


def get_tokenizer_for_file(filename: str) -> tuple[str, str]:
    """Return the corresponding tokenizer vocab and merges file paths based on filename"""
    # Get the path to tokenizer_models directory
    script_dir = os.path.dirname(__file__)
    tokenizer_models_dir = os.path.join(script_dir, "..", "tokenizer_models")
    
    if "owt" in filename.lower():
        vocab_file = os.path.join(tokenizer_models_dir, "tokenizer_vocab_owt_32k.json")
        merges_file = os.path.join(tokenizer_models_dir, "tokenizer_merges_owt_32k.txt")
    elif "tinystories" in filename.lower():
        vocab_file = os.path.join(tokenizer_models_dir, "tokenizer_vocab_tinystories_10k.json")
        merges_file = os.path.join(tokenizer_models_dir, "tokenizer_merges_tinystories_10k.txt")
    else:
        # Default to OWT tokenizer
        vocab_file = os.path.join(tokenizer_models_dir, "tokenizer_vocab_owt_32k.json")
        merges_file = os.path.join(tokenizer_models_dir, "tokenizer_merges_owt_32k.txt")
        print(f"Warning: Using default OWT tokenizer for file {filename}")
    
    return vocab_file, merges_file


def read_file_in_chunks(filepath: str, chunk_size: int = 8192) -> Iterator[str]:
    """Read large file in chunks"""
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def encode_file(input_file: str, output_file: str, chunk_size: int = 8192, max_tokens: int = None):
    """Encode a single file"""
    print(f"Processing file: {input_file}")
    
    # Get the corresponding tokenizer
    vocab_file, merges_file = get_tokenizer_for_file(input_file)
    print(f"Using tokenizer: {vocab_file}, {merges_file}")
    
    # Load tokenizer with special tokens
    # very important to include <|endoftext|> in the special tokens
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, ["<|endoftext|>"])
    
    # Get file size for progress display
    file_size = os.path.getsize(input_file)
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Encode file
    all_tokens = []
    processed_bytes = 0
    token_count = 0
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding") as pbar:
        for chunk in read_file_in_chunks(input_file, chunk_size):
            # Encode current chunk
            tokens = list(tokenizer.encode_iterable([chunk]))
            all_tokens.extend(tokens)
            
            # Update statistics
            processed_bytes += len(chunk.encode('utf-8'))
            token_count += len(tokens)
            pbar.update(len(chunk.encode('utf-8')))
            
            # Check if maximum token limit is reached
            if max_tokens and token_count >= max_tokens:
                print(f"Reached maximum token limit: {max_tokens}")
                all_tokens = all_tokens[:max_tokens]
                break
    
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Token compression ratio: {len(all_tokens) / (processed_bytes + 1e-8):.4f} tokens/byte")
    
    # Check token ID range to ensure it can be stored with uint16
    max_token_id = max(all_tokens) if all_tokens else 0
    min_token_id = min(all_tokens) if all_tokens else 0
    print(f"Token ID range: {min_token_id} - {max_token_id}")
    
    # Check for EOS tokens
    eos_count = sum(1 for token in all_tokens if token == 256)
    print(f"EOS tokens found: {eos_count} ({eos_count/len(all_tokens)*100:.4f}%)")
    
    if max_token_id > 65535:
        print(f"Warning: Maximum token ID ({max_token_id}) exceeds uint16 range (0-65535)")
        print("Consider using uint32 instead of uint16")
        dtype = np.uint32
    else:
        dtype = np.uint16
    
    # Convert to NumPy array and save
    print(f"Converting to {dtype} array...")
    token_array = np.array(all_tokens, dtype=dtype)
    
    print(f"Saving to {output_file}...")
    np.save(output_file, token_array)
    
    # Print final statistics
    array_size_mb = token_array.nbytes / (1024**2)
    print(f"Saved array size: {array_size_mb:.2f} MB")
    print(f"Array shape: {token_array.shape}")
    print(f"Array dtype: {token_array.dtype}")
    
    return token_array


def main():
    parser = argparse.ArgumentParser(description="Encode text files using BPE tokenizer")
    parser.add_argument("--input", "-i", 
                        help="Input file path (if not specified, process all files in data/)")
    parser.add_argument("--output", "-o",
                        help="Output directory (default: ../tokenized_data/)")
    parser.add_argument("--data-dir", "-d", default="../../data", 
                        help="Data directory containing text files (default: ../../data)")
    parser.add_argument("--chunk-size", type=int, default=8192, 
                        help="Chunk size for reading files (default: 8192)")
    parser.add_argument("--max-tokens", type=int, 
                        help="Maximum number of tokens to encode per file")
    parser.add_argument("--list", action="store_true", 
                        help="List available data files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output or "../tokenized_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # List available files
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found")
        return
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if args.list:
        print("Available data files:")
        for f in data_files:
            file_path = os.path.join(data_dir, f)
            size_gb = os.path.getsize(file_path) / (1024**3)
            print(f"  {f} ({size_gb:.2f} GB)")
        return
    
    # Process files
    if args.input:
        # Process specified file
        if not os.path.exists(args.input):
            print(f"Error: File '{args.input}' not found")
            return
        
        input_file = args.input
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, base_name + "_tokens.npy")
        
        encode_file(input_file, output_file, args.chunk_size, args.max_tokens)
    
    else:
        # Process all .txt files in data directory
        print(f"Processing all .txt files in {data_dir}/")
        print(f"Found {len(data_files)} files")
        
        for filename in data_files:
            input_file = os.path.join(data_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, base_name + "_tokens.npy")
            
            print(f"\n{'='*50}")
            try:
                encode_file(input_file, output_file, args.chunk_size,
                            args.max_tokens)
                print(f"✅ Successfully processed {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
        
        print(f"\n{'='*50}")
        print("All files processed!")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
