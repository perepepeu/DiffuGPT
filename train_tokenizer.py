# train_tokenizer.py

from tokenizer import BPEFastTokenizer

def main():
    tok = BPEFastTokenizer()

    with open("data/dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("Treinando tokenizer...")
    tok.train(text, vocab_size=1024)

    tok.save("data/tok-1024.model")
    print("Tokenizer salvo em data/tok-1024.model")

if __name__ == "__main__":
    main()