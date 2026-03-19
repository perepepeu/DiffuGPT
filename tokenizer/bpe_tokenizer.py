# tokenizer/bpe_tokenizer.py

import json
from collections import Counter

import regex as re


class BPEFastTokenizer:
    def __init__(self):
        # Vocabulário base: cada token inicial representa um byte único (0..255).
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Mapeia pares de tokens mesclados para novos IDs aprendidos no treino.
        self.merges = {}

        # Tamanho atual do vocabulário.
        self.vocab_size = len(self.vocab)

        # Regex de pré-tokenização.
        # Separa texto em pedaços preservando letras, números, espaços e símbolos,
        # com suporte a Unicode por meio de \p{L} e \p{N}.
        self.regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""

    def _reset(self):
        # Reinicia o tokenizer para o estado base antes de um novo treinamento.
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.vocab_size = len(self.vocab)

    def train(self, text: str, vocab_size: int):
        # Valida os parâmetros básicos de entrada.
        if not isinstance(text, str):
            raise TypeError("text precisa ser str")
        if vocab_size < 256:
            raise ValueError("vocab_size precisa ser pelo menos 256")

        # Garante treino limpo caso o mesmo objeto seja reutilizado.
        self._reset()

        # Pré-tokeniza o texto em unidades menores e converte cada pedaço para bytes.
        words = re.findall(self.regex, text)
        word_bytes = [tuple(word.encode("utf-8")) for word in words]

        # Conta frequência de cada sequência de bytes no corpus.
        word_freq = Counter(word_bytes)

        # Número de merges que precisam ser aprendidos além do vocabulário base.
        num_merges = vocab_size - 256

        for _ in range(num_merges):
            pair_freq = Counter()

            # Conta a frequência de cada par adjacente no corpus atual.
            for wb, freq in word_freq.items():
                for i in range(len(wb) - 1):
                    pair = (wb[i], wb[i + 1])
                    pair_freq[pair] += freq

            # Encerra caso não existam mais pares para mesclar.
            if not pair_freq:
                break

            # Seleciona o par mais frequente para criar um novo token.
            best_pair = max(pair_freq, key=pair_freq.get)
            new_id = len(self.vocab)

            # Registra o merge e cria a representação em bytes do novo token.
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Reescreve o corpus substituindo ocorrências do melhor par pelo novo token.
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

                new_word_freq[tuple(new_wb)] += freq

            word_freq = new_word_freq

        # Atualiza o tamanho final do vocabulário após o treinamento.
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        # Valida a entrada.
        if not isinstance(text, str):
            raise TypeError("text precisa ser str")

        # Pré-tokeniza e converte cada pedaço em bytes UTF-8.
        words = re.findall(self.regex, text)
        word_bytes = [tuple(word.encode("utf-8")) for word in words]

        output_ids = []

        # Para cada pedaço, aplica os merges aprendidos até não haver mais mudanças.
        for wb in word_bytes:
            ids = list(wb)

            changed = True
            while changed:
                changed = False
                new_ids = []
                i = 0

                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) in self.merges:
                        new_ids.append(self.merges[(ids[i], ids[i + 1])])
                        i += 2
                        changed = True
                    else:
                        new_ids.append(ids[i])
                        i += 1

                ids = new_ids

            output_ids.extend(ids)

        return output_ids

    def decode(self, ids: list[int]) -> str:
        # Aceita qualquer iterável de IDs, convertendo para lista se necessário.
        if not isinstance(ids, list):
            ids = list(ids)

        # Reconstrói a sequência de bytes correspondente aos tokens.
        all_bytes = b""
        for token_id in ids:
            token_id = int(token_id)
            if token_id not in self.vocab:
                raise ValueError(f"token id {token_id} não existe no vocab")
            all_bytes += self.vocab[token_id]

        # Tenta decodificar como UTF-8; se falhar, usa fallback mais permissivo.
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return all_bytes.decode("latin1", errors="replace")

    def save(self, path: str):
        # Serializa o vocabulário em hex para tornar o JSON compatível com bytes.
        vocab_json = {str(k): v.hex() for k, v in self.vocab.items()}

        # Serializa merges usando "id1,id2" como chave textual.
        merges_json = {f"{k[0]},{k[1]}": int(v) for k, v in self.merges.items()}

        data = {
            "version": 1,
            "regex": self.regex,
            "vocab_size": self.vocab_size,
            "vocab": vocab_json,
            "merges": merges_json,
        }

        # Salva o tokenizer treinado em formato JSON.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        # Lê o arquivo salvo do tokenizer.
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Valida os campos essenciais do arquivo.
        if "vocab" not in data or "merges" not in data:
            raise ValueError("Arquivo de tokenizer inválido: faltam campos 'vocab' ou 'merges'")

        # Reconstrói o vocabulário a partir do formato salvo em hex.
        vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}

        # Reconstrói o dicionário de merges a partir das chaves serializadas.
        merges = {}
        for k, v in data["merges"].items():
            k0, k1 = map(int, k.split(","))
            merges[(k0, k1)] = int(v)

        self.vocab = vocab
        self.merges = merges
        self.vocab_size = len(self.vocab)

        # Reaproveita a regex salva no arquivo, se disponível.
        if "regex" in data and isinstance(data["regex"], str):
            self.regex = data["regex"]
