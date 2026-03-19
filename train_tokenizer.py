# train_tokenizer.py

from tokenizer import BPEFastTokenizer


def main():
    # Define os caminhos de entrada e saída, além do tamanho desejado do vocabulário.
    dataset_path = "data/dataset.txt"
    output_path = "data/tok-1024.model"
    vocab_size = 1024

    # Lê todo o conteúdo do dataset que será usado no treinamento do tokenizer.
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Cria uma instância do tokenizer BPE.
    tok = BPEFastTokenizer()

    # Inicia o treinamento do tokenizer com o texto carregado.
    print("Treinando tokenizer...")
    tok.train(text, vocab_size=vocab_size)

    # Salva o tokenizer treinado no arquivo de saída.
    tok.save(output_path)

    # Exibe informações finais para conferência do treinamento.
    print(f"Tokenizer salvo em {output_path}")
    print(f"Vocab final: {tok.vocab_size} tokens")
    print(f"Merges aprendidos: {len(tok.merges)}")


# Executa o treinamento apenas quando o arquivo for rodado diretamente.
if __name__ == "__main__":
    main()
