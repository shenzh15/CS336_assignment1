import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool
import heapq
import json

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

def pop_best_pair(heap, version, pair_cnt):
    """Find the best pair to merge from the heap"""
    while heap:
        neg_f, ver, p = heapq.heappop(heap)
        if version[p] == ver and -neg_f == pair_cnt[p] and neg_f != 0:   # valid
            freq_top = -neg_f
            bucket = [p]

            # Also temporarily extract other pairs with the same frequency from the heap
            while heap and -heap[0][0] == freq_top:
                n2, v2, p2 = heapq.heappop(heap)
                if version[p2] == v2 and -n2 == pair_cnt[p2]:
                    bucket.append(p2)

            # best = sorted(bucket, key=lambda x: (x[0], x[1]), reverse=True)[0]
            best = max(bucket)        # lexicographically largest
            for other in bucket:
                if other is not best:
                    heapq.heappush(heap, (-pair_cnt[other], version[other], other))

            return best
    return None      # heap is empty

def apply_bpe_merge(best_pair, heap, version, pair_cnt, tokens):
    """Execute BPE merge operation"""
    a, b = best_pair
    ab = a + b
    pair_cnt[best_pair] = 0
    version[best_pair] -= 1    # invalidate old entries

    # Record sequences to delete
    to_delete = set()
    # Record sequences to add
    to_add = {}

    for seq, f in tokens.items():
        toks = list(seq)
        i = 0
        modified = False
        while i < len(toks) - 1:
            if toks[i] == a and toks[i + 1] == b:
                # ---- Decrease old counts ---------------------
                if i:
                    left = (toks[i - 1], a)
                    pair_cnt[left] -= f
                    version[left] = version.get(left, 0) - 1
                    heapq.heappush(heap, (-pair_cnt[left], version[left], left))
                if i + 2 < len(toks):
                    right = (b, toks[i + 2])
                    pair_cnt[right] -= f
                    version[right] = version.get(right, 0) - 1
                    heapq.heappush(heap, (-pair_cnt[right], version[right], right))

                # ---- Merge itself ---------------------
                toks[i:i + 2] = [ab]
                modified = True

                if i:
                    left = (toks[i - 1], ab)
                    pair_cnt[left] += f
                    version[left] = version.get(left, 0) - 1
                    heapq.heappush(heap, (-pair_cnt[left], version[left], left))
                if i + 1 < len(toks):
                    right = (ab, toks[i + 1])
                    pair_cnt[right] += f
                    version[right] = version.get(right, 0) - 1
                    heapq.heappush(heap, (-pair_cnt[right], version[right], right))
            else:
                i += 1

        if modified:
            to_delete.add(seq)
            new_seq = tuple(toks)
            to_add[new_seq] = to_add.get(new_seq, 0) + f

    # Execute delete and add operations
    for seq in to_delete:
        del tokens[seq]
    tokens.update(to_add)

    return tokens, ab

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
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes,
                                           "<|endoftext|>".encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
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
        tokens = byte_tuple_freq

        pair_cnt = defaultdict(int)
        # pair_cnt = Counter()
        for seq, f in tokens.items():
            for i in range(len(seq) - 1):
                pair_cnt[(seq[i], seq[i + 1])] += f

        # Heap elements: (-freq, version, pair); version is used for lazy invalidation
        version = {p: 0 for p in pair_cnt}
        heap = [(-c, 0, p) for p, c in pair_cnt.items()]
        heapq.heapify(heap)

        merged = []
        i_vocab_size = len(vocab)
        while i_vocab_size < vocab_size:
            best_pair = pop_best_pair(heap, version, pair_cnt)
            if not best_pair or pair_cnt[best_pair] == 0:
                break
            tokens, new_tok = apply_bpe_merge(best_pair, heap, version,
                                              pair_cnt, tokens)
            id_to_token[i_vocab_size] = new_tok
            i_vocab_size += 1
            merged.append(best_pair)

        return id_to_token, merged

if __name__ == "__main__":
    id_to_token, merged = train_bpe("./data/haha.txt", 300, ["<|endoftext|>"])
    # id_to_token, merged = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])

    with open("tokenizer_vocab_haha.json", "w", encoding="utf-8") as f:
        json.dump({
            str(k): v.hex() for k, v in id_to_token.items()
        }, f, indent=2)
    with open("tokenizer_merges_haha.tsv", "w", encoding="utf-8") as f:
        for a, b in merged:
            f.write(f"{a.hex()}\t{b.hex()}\n")