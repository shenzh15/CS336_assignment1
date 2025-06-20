import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool
import time
import json
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def word_to_byte_tuple(word):
    b = word.encode("utf-8")
    return tuple(b[i:i+1] for i in range(len(b)))

def apply_bpe_merge(best_pair, pair_cnt, byte_tuple_freq, pair_occurrences):
    """Execute BPE merge operation"""
    a, b = best_pair
    ab = a + b
    pair_cnt.pop(best_pair, None)

    affected_seqs = pair_occurrences.pop(best_pair, set())

    to_delete = []
    to_add = {}

    pair_deltas = {}

    for seq in affected_seqs:
        f = byte_tuple_freq[seq]
        toks = list(seq)
        
        modified = False

        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i + 1])
            if pair in pair_occurrences:  # Avoid creating empty sets in defaultdict
                pair_occurrences[pair].discard(seq)
        
        new_toks = []
        i = 0
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                # ---- Decrease old counts ---------------------
                if new_toks:
                    left = (new_toks[-1], a)
                    pair_deltas[left] = pair_deltas.get(left, 0) - f
                if i + 2 < len(toks):
                    right = (b, toks[i + 2])
                    pair_deltas[right] = pair_deltas.get(right, 0) - f

                # ---- Merge itself ---------------------
                new_toks.append(ab)
                modified = True

                if len(new_toks) > 1:
                    left = (new_toks[-2], ab)
                    pair_deltas[left] = pair_deltas.get(left, 0) + f
                if i + 2 < len(toks):
                    right = (ab, toks[i + 2])
                    pair_deltas[right] = pair_deltas.get(right, 0) + f
                i += 2  # Skip merged token
            else:
                new_toks.append(toks[i])
                i += 1

        if modified:
            to_delete.append(seq)
            to_add[tuple(new_toks)] = f  

    # update pair_cnt
    for pair, delta in pair_deltas.items():
        if delta:
            current = pair_cnt.get(pair, 0)
            new_count = current + delta
            if new_count <= 0:
                pair_cnt.pop(pair, None)
            else:
                pair_cnt[pair] = new_count

    # update byte_tuple_freq
    for seq in to_delete:
        del byte_tuple_freq[seq]
    byte_tuple_freq.update(to_add)

    # update pair_occurrences
    for new_seq in to_add:
        for i in range(len(new_seq) - 1):
            pair_occurrences[(new_seq[i], new_seq[i + 1])].add(new_seq)

    return byte_tuple_freq, ab

def pre_tokenize_chunk(args):
    start, end, input_path, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")

    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        flags=re.VERSION1
    )
    SEP_RE = re.compile("|".join(map(re.escape, special_tokens)))
    word_freq = defaultdict(int)
    for sentence in SEP_RE.split(chunk):
        for word in PAT.finditer(sentence):
            word_freq[word.group()] += 1

    return word_freq

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    show_progress = kwargs.get('show_progress', False)

    with open(input_path, "rb") as f:
        num_processes = min(20, os.cpu_count() or 1)
        boundaries = find_chunk_boundaries(f, num_processes,
                                           "<|endoftext|>".encode("utf-8"))

        args_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            args_list.append((start, end, input_path, special_tokens))

        with Pool(len(boundaries) - 1) as pool:
            results = pool.map(pre_tokenize_chunk, args_list)

        global_word_freq = Counter()
        for wc in results:
            global_word_freq.update(wc)

        # initialize vocab
        basic_byte_tokens = [bytes([i]) for i in range(256)]
        special_bytes_tokens = [
            token.encode("utf-8") for token in special_tokens
        ]
        vocab = basic_byte_tokens + special_bytes_tokens
        id_to_token = {idx: tok for idx, tok in enumerate(vocab)}

        byte_tuple_freq = {
            word_to_byte_tuple(word): c
            for word, c in global_word_freq.items()
        }

        pair_occurrences = defaultdict(set)
        pair_cnt = defaultdict(int)
        for seq, f in byte_tuple_freq.items():
            for i in range(len(seq) - 1):
                pair_cnt[(seq[i], seq[i + 1])] += f
                pair_occurrences[(seq[i], seq[i + 1])].add(seq)
        
        merged = []
        i_vocab_size = len(vocab)
        
        if show_progress:
            progress_bar = tqdm(total=vocab_size - i_vocab_size, 
                                desc="Training BPE", 
                                unit="merges")
        
        while i_vocab_size < vocab_size:
            max_freq = max(pair_cnt.values())
            candidates = [pair for pair, freq in pair_cnt.items() if freq == max_freq]
            best_pair = max(candidates)  # lexicographically largest
            if not best_pair:
                break
            byte_tuple_freq, new_tok = apply_bpe_merge(best_pair,
                                              pair_cnt, byte_tuple_freq, pair_occurrences)
            id_to_token[i_vocab_size] = new_tok
            i_vocab_size += 1
            merged.append(best_pair)
            
            if show_progress:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'vocab_size': i_vocab_size,
                    'max_freq': max_freq
                })
        
        if show_progress:
            progress_bar.close()

        return id_to_token, merged

if __name__ == "__main__":
    id_to_token, merged = train_bpe("./data/owt_train.txt", 32000, ["<|endoftext|>"], show_progress=True)
    # id_to_token, merged = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])

    with open("tokenizer_vocab_owt_32k.json", "w", encoding="utf-8") as f:
        json.dump({
            str(k): v.hex() for k, v in id_to_token.items()
        }, f, indent=2)
    with open("tokenizer_merges_owt_32k.txt", "w", encoding="utf-8") as f:
        for a, b in merged:
            f.write(f"{a.hex()}\t{b.hex()}\n")