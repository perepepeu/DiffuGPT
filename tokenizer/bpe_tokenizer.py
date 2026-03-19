# tokenizer/bpe_tokenizer.py

import regex as re
import json
import os
from collections import Counter


class BPEFastTokenizer:
    def __init__(self):
        # lista bytes iniciais 0–255
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        # regex para pré‑tokenizar em "words"
        self.regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^ \p{L}\p{N}]+| \n (?! )| \n"""

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256, "vocab_size precisa ser pelo menos 256 (bytes)"

        # pré‑tokenizar em "words"
        words = re.findall(self.regex, text)
        word_bytes = [tuple(word.encode("utf-8")) for word in words]
        word_freq = Counter(word_bytes)

        # loop de merges
        num_merges = vocab_size - 256
        for _ in range(num_merges):
            pair_freq = Counter()
            for wb, freq in word_freq.items():
                for i in range(len(wb) - 1):
                    pair = (wb[i], wb[i + 1])
                    pair_freq[pair] += freq

            if len(pair_freq) == 0:
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            new_id = len(self.vocab)
            self.merges[best_pair] = new_id

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_word_freq = Counter()
            for wb, freq in word_freq.items():
                new_wb = []
                i = 0
                while i < len(wb):
                    if i < len(wb) - 1 and (wb[i], wb[i + 1]) == best_pair:
                        new_wb.append(new_id)
                        i += 2
                    else:
                        new_wb.append(wb[i])
                        i += 1
                new_word_freq[tuple(new_wb)] = freq
            word_freq = new_word_freq

    def encode(self, text: str) -> list[int]:
        words = re.findall(self.regex, text)
        word_bytes = [tuple(word.encode("utf-8")) for word in words]

        word_ids = []
        for wb in word_bytes:
            ids = list(wb)
            changed = True
            while changed:
                changed = False
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) in self.merges:
                        new_id = self.merges[(ids[i], ids[i + 1])]
                        new_ids.append(new_id)
                        i += 2
                        changed = True
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            word_ids.extend(ids)
        return word_ids

    def decode(self, ids: list[int]) -> str:
        all_bytes = b""
        for id_ in ids:
            if id_ not in self.vocab:
                raise ValueError(f"token id {id_} não existe no vocab")
            all_bytes += self.vocab[id_]
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return all_bytes.decode("latin1", errors="replace")

    def save(self, path: str):
        vocab_json = {str(k): v.hex() for k, v in self.vocab.items()}
        merges_json = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {"vocab": vocab_json, "merges": merges_json}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        merges = {}
        for k, v in data["merges"].items():
            k0, k1 = map(int, k.split(","))
            merges[(k0, k1)] = v
        self.vocab = vocab
        self.merges = merges
