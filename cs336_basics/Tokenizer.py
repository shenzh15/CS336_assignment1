from typing import Iterable, Iterator
import json
import regex as re
from collections import defaultdict

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.special_tokens = special_tokens or []
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
    
    def encode(self, text: str) -> list[int]:
        PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=re.VERSION1
            )
        SEP_RE = re.compile("|".join(map(re.escape, self.special_tokens)))
        if self.special_tokens:
            SEP_RE = re.compile("|".join(map(re.escape, self.special_tokens)))
            segments = SEP_RE.split(text)
            matches = list(SEP_RE.finditer(text))
        else:
            segments = [text] # if no special tokens, split the text into a single sentence
            matches = []
        
        result = []
        seg_idx = 0
        match_idx = 0

        while seg_idx < len(segments) or match_idx < len(matches):
            if seg_idx < len(segments):
                segment = segments[seg_idx]
                if segment:
                    for word in PAT.finditer(segment):
                        bytes_word = word.group().encode("utf-8")
                        tokens = [bytes_word[i:i+1] for i in range(len(bytes_word))]
                        while True:
                            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
                            ranked = [(pair, self.merge_ranks[pair]) for pair in pairs if pair in self.merge_ranks]
                            if not ranked:
                                break
                            best_pair = min(ranked, key=lambda x: x[1])[0]
                            i = 0
                            while i < len(tokens) - 1:
                                if (tokens[i], tokens[i+1]) == best_pair:
                                    tokens[i:i+2] = [tokens[i] + tokens[i+1]]
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

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass 
    
    def decode(self, ids: list[int]) -> str:
        pass 
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
            vocab = {
                int(k): bytes.fromhex(v) for k, v in vocab_raw.items()
            }
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [
                (bytes.fromhex(a), bytes.fromhex(b))
                for line in f
                for a, b in [line.strip().split("\t")]
            ]
        return cls(vocab, merges, special_tokens)