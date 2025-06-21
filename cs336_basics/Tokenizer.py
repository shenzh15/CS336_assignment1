from typing import Iterable, Iterator
import json
import regex as re
# from collections import defaultdict


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        # self.special_tokens = special_tokens or []
        self.special_tokens = sorted(special_tokens or [],
                                     key=len,
                                     reverse=True)
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.SEP_RE = re.compile("|".join(map(re.escape, self.special_tokens)))
        self.PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=re.VERSION1)

    def _segment_and_match_iter(self, text: str) -> Iterator[tuple[str, str]]:
        if not self.special_tokens:
            yield text, None
            return

        last_end = 0
        for match in self.SEP_RE.finditer(text):
            start, end = match.span()
            yield text[last_end:start], match.group()
            last_end = end
        yield text[last_end:], None  # 最后一个 segment 没有 match 跟随

    def _apply_bpe_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of byte tokens."""
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            ranked = [(pair, self.merge_ranks[pair]) for pair in pairs if pair in self.merge_ranks]
            if not ranked:
                break
            
            best_pair = min(ranked, key=lambda x: x[1])[0]
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == best_pair:
                    tokens[i:i + 2] = [tokens[i] + tokens[i + 1]]
                    break
                i += 1
        return tokens

    def _process_word(self, word_match) -> Iterator[int]:
        """Process a single word match and yield token IDs."""
        bytes_word = word_match.group().encode("utf-8")
        tokens = [bytes_word[i:i + 1] for i in range(len(bytes_word))]
        tokens = self._apply_bpe_merges(tokens)
        
        for tok in tokens:
            yield self.token_to_id[tok]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for segment, match in self._segment_and_match_iter(text):
                # Process regular text in the segment
                for word in self.PAT.finditer(segment):
                    yield from self._process_word(word)
                
                # Process special token if present
                if match:
                    yield self.token_to_id[match.encode("utf-8")]
    
    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    # ------------deprecated old method------------
    def encode_iterable_old(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode_old(text)
    
    def encode_old(self, text: str) -> list[int]:
        if self.special_tokens:
            segments = self.SEP_RE.split(text)
            matches = list(self.SEP_RE.finditer(text))
        else:
            segments = [
                text
            ]  # if no special tokens, split the text into a single sentence
            matches = []

        result = []
        seg_idx = 0
        match_idx = 0

        while seg_idx < len(segments) or match_idx < len(matches):
            if seg_idx < len(segments):
                segment = segments[seg_idx]
                if segment:
                    for word in self.PAT.finditer(segment):
                        bytes_word = word.group().encode("utf-8")
                        tokens = [
                            bytes_word[i:i + 1] for i in range(len(bytes_word))
                        ]
                        while True:
                            pairs = [(tokens[i], tokens[i + 1])
                                     for i in range(len(tokens) - 1)]
                            ranked = [(pair, self.merge_ranks[pair])
                                      for pair in pairs
                                      if pair in self.merge_ranks]
                            if not ranked:
                                break
                            best_pair = min(ranked, key=lambda x: x[1])[0]
                            i = 0
                            while i < len(tokens) - 1:
                                if (tokens[i], tokens[i + 1]) == best_pair:
                                    tokens[i:i +
                                           2] = [tokens[i] + tokens[i + 1]]
                                    break
                                i += 1
                        for tok in tokens:
                            result.append(self.token_to_id[tok])
                seg_idx += 1
            if match_idx < len(matches):
                special_token = matches[match_idx].group().encode("utf-8")
                result.append(self.token_to_id[special_token])
                match_idx += 1
        return result
    # ------------deprecated old method------------

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                # Handle unknown token IDs gracefully
                raise ValueError(f"Unknown token ID: {id}")

        # Concatenate all byte tokens and decode to string
        combined_bytes = b''.join(tokens)
        return combined_bytes.decode('utf-8', errors='replace')

    @classmethod
    def from_files(cls,
                   vocab_filepath,
                   merges_filepath,
                   special_tokens=None) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
            vocab = {int(k): bytes.fromhex(v) for k, v in vocab_raw.items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for line in f
                      for a, b in [line.strip().split("\t")]]
        return cls(vocab, merges, special_tokens)
