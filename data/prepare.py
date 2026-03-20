# data/prepare.py

import numpy as np
from tokenizer import BPEFastTokenizer
from config import Config


def main():
    cfg = Config()
    tok = BPEFastTokenizer()
    tok.load("data/tok-1024.model")

    with open("data/dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()

    ids = tok.encode(text)

    block_size = cfg.block_size

    arr = np.array(ids, dtype=np.int32)
    n = len(arr) // block_size
    arr = arr[: n * block_size]

    arr = arr.reshape(n, block_size)

    np.save("data/train_ids.npy", arr)
    print("Dataset pronto:", arr.shape)


if __name__ == "__main__":
    main()
