# DiffuGPT

> Hybrid discrete diffusion + autoregressive text generation, built from scratch.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)]()


Modelo de linguagem híbrido construído do zero em Python/PyTorch que combina **geração por difusão discreta** com **refinamento autoregressivo (GPT-style)**.

---

## 📐 Arquitetura

O modelo opera em dois modos com um **único Transformer compartilhado**:

| Modo | Atenção | Objetivo |
|---|---|---|
| **Diffusion** | Bidirecional (sem máscara) | Prever tokens mascarados iterativamente |
| **AR Refiner** | Causal (máscara triangular) | Prever próximo token (GPT-style) |

Os embeddings de token, posição e a `lm_head` são compartilhados entre os dois modos.

### Pipeline de inferência híbrida

```
Prompt
  │
  ▼
[Diffusion — bidirecional, t=max_t → t=1]
  │   gera rascunho global
  ▼
[AR Refiner — causal, 1 forward pass]
  │   corrige tokens com baixa confiança
  ▼
Texto final
```

---

## 📁 Estrutura do projeto

```
DiffuGPT/
│   config.py            # Hiperparâmetros e configurações
│   train.py             # Loop de treinamento híbrido
│   train_tokenizer.py   # Treina o tokenizer BPE
│   sample.py            # Geração via CLI (3 modos)
│   requirements.txt
│
├───data/
│   │   dataset.txt      # Corpus de texto bruto ⚠️ não apagar
│   │   dataset.py       # Dataset PyTorch (carrega train_ids.npy)
│   │   prepare.py       # Tokeniza dataset e salva como .npy
│   │   tok-1024.model   # Tokenizer treinado (gerado)
│   │   train_ids.npy    # Dataset tokenizado (gerado)
│
├───model/
│   │   hybrid_model.py  # HybridModel — forward_diffusion + forward_ar
│   │   scheduler.py     # NoiseScheduler — mascaramento com schedule cosseno
│
├───tokenizer/
│   │   bpe_tokenizer.py # BPE tokenizer implementado do zero
│   │   __init__.py
│
├───web/
│   │   app.py           # API Flask (modos: diffusion / ar / hybrid)
│   ├───static/
│   │       style.css
│   └───templates/
│           index.html
│
└───checkpoints/         # Checkpoints por época (gerados no treino)
```

---

## 📂 Formato do dataset

O modelo espera um único arquivo de texto plano em `data/dataset.txt`.

### Requisitos

| Item | Detalhe |
|---|---|
| Formato | `.txt` — texto puro, UTF-8 |
| Tamanho mínimo recomendado | ~500 KB (quanto maior, melhor) |
| Idioma | Qualquer — o tokenizer BPE aprende do próprio corpus |
| Estrutura | Texto corrido, sem formatação especial obrigatória |

### Exemplos de fontes válidas

- Livros e contos em `.txt` (ex: Project Gutenberg)
- Artigos exportados da Wikipedia
- Letras de músicas / poesias concatenadas
- Diálogos e roteiros
- Qualquer corpus de texto no idioma que quiser gerar

### Exemplo do formato esperado

```
Era uma vez um reino muito distante onde todos viviam em paz.
O rei era justo e a rainha, sábia. Juntos governavam com bondade.
Certo dia, um viajante chegou trazendo notícias do além-mar...
```

> Não é necessário separar por frases ou parágrafos de forma especial.  
> O `prepare.py` tokeniza tudo e divide automaticamente em blocos de `block_size=128` tokens.

### Como o pipeline processa

```
dataset.txt
    │  train_tokenizer.py  → aprende 1024 merges BPE sobre o corpus
    ▼
tok-1024.model
    │  data/prepare.py     → tokeniza e divide em blocos de 128 tokens
    ▼
train_ids.npy              → array (N, 128) pronto para o DataLoader
```

---

## ⚙️ Configuração (`config.py`)

| Parâmetro | Valor | Descrição |
|---|---|---|
| `vocab_size` | 1024 | Tamanho do vocabulário BPE |
| `block_size` | 128 | Comprimento máximo da sequência |
| `emb_dim` | 256 | Dimensão dos embeddings |
| `n_layers` | 6 | Número de camadas Transformer |
| `n_heads` | 8 | Número de cabeças de atenção |
| `max_timesteps` | 8 | Passos de difusão |
| `ar_alpha` | 0.5 | Peso da loss AR (0=só diffusion, 1=só AR) |
| `ar_refine_mode` | `"balanced"` | Modo do refinador: `fast` ou `balanced` |
| `ar_threshold_fast` | 0.65 | Threshold de confiança — modo fast |
| `ar_threshold_balanced` | 0.55 | Threshold de confiança — modo balanced |

---

## 🚀 Como usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o tokenizer

```bash
python train_tokenizer.py
```

### 3. Preparar o dataset

```bash
python data/prepare.py
```

### 4. Treinar o modelo

```bash
python train.py
```

Acompanhe as métricas em tempo real no terminal:
```
Epoch 1/15 | Train 5.67 | Val 5.22 | Diff 5.84 | AR 5.50
```

### 5. Gerar texto via CLI

```bash
python sample.py
```

Modos disponíveis:
```
1) Diffusion pura
2) AR puro (GPT-style)
3) Híbrido — balanced  (padrão)
4) Híbrido — fast      (velocidade máxima)
```

### 6. Interface Web

```bash
cd web
python app.py
```

Acesse `http://localhost:5000` no navegador.

#### Endpoint `/generate`

```json
POST /generate
{
  "prompt": "Era uma vez",
  "mode": "hybrid",
  "min_tokens": 32,
  "temperature": 1.0,
  "threshold": 0.55
}
```

Modos aceitos: `"diffusion"`, `"ar"`, `"hybrid"`

---

## 🔁 Fluxo completo do zero

```bash
# 1. treinar tokenizer
python train_tokenizer.py

# 2. processar dataset
python data/prepare.py

# 3. treinar modelo
python train.py

# 4. gerar texto
python sample.py
```

---

## 🧹 Limpar arquivos gerados

```powershell
Remove-Item -Force model.pt -ErrorAction SilentlyContinue
Remove-Item -Force logs.json -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force checkpoints\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force loss\ -ErrorAction SilentlyContinue
Remove-Item -Force data\train_ids.npy -ErrorAction SilentlyContinue
Remove-Item -Force data\tok-1024.model -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## 📦 Dependências (`requirements.txt`)

```
torch
flask
tqdm
numpy
regex
```

---

## 📊 Treinamento híbrido

O loop de treino calcula **duas losses por batch** e as combina:

```python
loss = (1 - ar_alpha) * loss_diffusion + ar_alpha * loss_ar
```

A loss AR é calculada a cada 2 steps para reduzir o tempo de treino em ~30% sem impacto significativo na qualidade.

---

## 🤖 Sobre o projeto

- **Tokenizer BPE** implementado do zero (sem HuggingFace)
- **Difusão discreta** com schedule cosseno de mascaramento
- **Transformer bidirecional** para difusão e causal para AR — pesos compartilhados
- **Interface web** Flask com suporte aos 3 modos de geração
