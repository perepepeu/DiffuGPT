# DiffuGPT

> Geração de texto híbrida por difusão discreta + refinamento autoregressivo, construída do zero em Python e PyTorch.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)]()

O **DiffuGPT** é um modelo de linguagem híbrido construído do zero que combina **difusão discreta mascarada** para gerar um rascunho global com **refinamento autoregressivo estilo GPT** para corrigir tokens, melhorar coerência e aumentar a fluidez final.

Ele foi pensado como um projeto de pesquisa prático, leve e modificável, com pipeline completo de:
- tokenização;
- preparação de dados;
- treinamento;
- geração por CLI;
- interface web.

---

## Destaques

- **Construído do zero** — sem classes prontas de modelos da Hugging Face.
- **Arquitetura híbrida** — difusão para estrutura global, AR para refinamento local.
- **Tokenizer próprio** — BPE byte-level implementado manualmente.
- **Transformer compartilhado** — o mesmo backbone opera em dois modos.
- **Pipeline completo** — dataset, tokenizer, treino, inferência e interface web no mesmo repositório.
- **Fácil de estudar e modificar** — ideal para pesquisa, portfólio e experimentação.
- **Geração flexível** — modo diffusion, AR puro ou híbrido.

---

## Arquitetura

O modelo opera em dois modos usando um **único Transformer compartilhado**:

| Modo | Atenção | Objetivo |
|---|---|---|
| **Diffusion** | Bidirecional | Prever tokens mascarados iterativamente |
| **AR Refiner** | Causal | Prever o próximo token no estilo GPT |

Componentes compartilhados:
- embeddings de token;
- embeddings posicionais;
- backbone Transformer;
- projeção de saída (`lm_head`).

Componente exclusivo do modo diffusion:
- embedding de timestep (`time_emb`).

### Pipeline de inferência híbrida

```text
Prompt
  │
  ▼
[Diffusion — denoising bidirecional]
  │   gera um rascunho global
  ▼
[AR Refiner — correção causal]
  │   corrige tokens fracos ou incoerentes
  ▼
Texto final
```

---

## Por que este projeto é interessante

O DiffuGPT explora uma abordagem diferente dos modelos autoregressivos tradicionais.

Em vez de depender apenas da geração token por token da esquerda para a direita, ele combina:
- **reconstrução global paralela** por difusão mascarada;
- **correção local sequencial** por geração autoregressiva;
- **um único backbone compartilhado** capaz de operar em dois comportamentos diferentes.

Isso torna o projeto interessante para:
- pesquisa em geração híbrida;
- estudo de difusão discreta em texto;
- experimentos com decodificação híbrida;
- aprendizado prático sobre tokenização, Transformers e treino de LMs;
- portfólio técnico diferenciado.

---

## Estrutura do projeto

```text
DiffuGPT/
│   config.py             # Hiperparâmetros e configuração geral
│   train.py              # Loop de treinamento híbrido
│   train_tokenizer.py    # Treina o tokenizer BPE
│   sample.py             # Geração de texto via CLI
│   requirements.txt
│   README.md
│
├── data/
│   │   dataset.txt       # Corpus bruto de treino
│   │   dataset.py        # Dataset PyTorch
│   │   prepare.py        # Tokeniza e salva blocos em .npy
│   │   tok-1024.model    # Tokenizer treinado
│   │   train_ids.npy     # Dataset tokenizado
│
├── model/
│   │   hybrid_model.py   # Transformer compartilhado com 2 modos
│   │   scheduler.py      # Scheduler cosseno de mascaramento
│
├── tokenizer/
│   │   bpe_tokenizer.py  # Tokenizer BPE implementado do zero
│   │   __init__.py
│
├── web/
│   │   app.py            # Interface/API Flask
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
└── checkpoints/          # Checkpoints gerados durante o treino
```

---

## Formato do dataset

O projeto espera um único arquivo de texto bruto em:

```text
data/dataset.txt
```

### Requisitos

| Item | Detalhe |
|---|---|
| Formato | `.txt` puro, UTF-8 |
| Tamanho mínimo recomendado | ~500 KB |
| Idioma | Qualquer idioma presente no corpus |
| Estrutura | Texto contínuo, sem formatação especial obrigatória |

### Fontes válidas

- livros e contos em `.txt`;
- exportações da Wikipedia;
- diálogos e conversas;
- letras de músicas e poesias;
- textos técnicos;
- corpus próprio de domínio específico.

### Exemplo

```text
Era uma vez um reino muito distante onde todos viviam em paz.
O rei era justo e a rainha, sábia. Juntos governavam com bondade.
Certo dia, um viajante chegou trazendo notícias do além-mar...
```

O pipeline faz automaticamente:
1. aprende o tokenizer BPE no corpus;
2. converte o texto em IDs de tokens;
3. divide a sequência em blocos fixos;
4. salva o resultado em `train_ids.npy` para treino.

---

## Pipeline de treinamento

```text
dataset.txt
   │  train_tokenizer.py   → treina o BPE byte-level
   ▼
tok-1024.model
   │  data/prepare.py      → tokeniza e divide em blocos
   ▼
train_ids.npy
   │  train.py             → treino híbrido (diffusion + AR)
   ▼
model.pt + checkpoints/
```

---

## Configuração

Principais parâmetros do `config.py`:

| Parâmetro | Valor | Descrição |
|---|---:|---|
| `vocab_size` | 1024 | Tamanho do vocabulário BPE |
| `block_size` | 128 | Comprimento máximo da sequência |
| `emb_dim` | 256 | Dimensão dos embeddings |
| `n_layers` | 6 | Número de camadas Transformer |
| `n_heads` | 8 | Número de cabeças de atenção |
| `max_timesteps` | 8 | Passos de difusão |
| `ar_alpha` | 0.5 | Peso da loss AR no treino híbrido |
| `ar_refine_mode` | `"balanced"` | Estratégia de refinamento AR |
| `ar_threshold_fast` | 0.65 | Threshold de confiança no modo fast |
| `ar_threshold_balanced` | 0.55 | Threshold de confiança no modo balanced |

---

## Recursos

- **Tokenizer BPE byte-level próprio**
- **Difusão discreta mascarada para texto**
- **Refinamento causal estilo GPT**
- **Backbone Transformer compartilhado**
- **Scheduler cosseno de ruído/mascaramento**
- **Treino híbrido com duas losses**
- **Geração interativa via CLI**
- **Interface web com Flask**
- **Checkpoints por época**
- **Código modular e legível**

---

## Instalação

```bash
pip install -r requirements.txt
```

### Dependências

```txt
torch
flask
tqdm
numpy
regex
```

---

## Início rápido

### 1) Treinar o tokenizer

```bash
python train_tokenizer.py
```

### 2) Preparar o dataset

```bash
python data/prepare.py
```

### 3) Treinar o modelo

```bash
python train.py
```

Exemplo de saída durante o treino:

```text
Epoch 1/15 | Train 5.67 | Val 5.22 | Diff 5.84 | AR 5.50
```

### 4) Gerar texto pela CLI

```bash
python sample.py
```

Modos disponíveis:

```text
1) Diffusion
2) AR puro (GPT-style)
3) Híbrido — balanced
4) Híbrido — fast
```

### 5) Executar a interface web

```bash
cd web
python app.py
```

Depois, acesse:

```text
http://localhost:5000
```

---

## Exemplo de API

### `POST /generate`

```json
{
  "prompt": "Era uma vez",
  "mode": "hybrid",
  "min_tokens": 32,
  "temperature": 1.0,
  "threshold": 0.55
}
```

Modos aceitos:
- `"diffusion"`
- `"ar"`
- `"hybrid"`

---

## Objetivo do treino híbrido

Cada batch combina dois sinais de aprendizado:

```python
loss = (1 - ar_alpha) * loss_diffusion + ar_alpha * loss_ar
```

Isso permite que o modelo aprenda:
- reconstrução global por difusão;
- previsão local token a token por modo autoregressivo.

---

## Pontos fortes do projeto

O que torna o DiffuGPT um projeto forte:

- **Não é apenas mais um GPT pequeno**; ele testa uma ideia híbrida real.
- **Explora difusão em texto**, que ainda é uma área muito menos comum que AR puro.
- **Une pesquisa e prática**, porque inclui tokenizer, treino, inferência e interface.
- **É bom para portfólio**, por mostrar domínio de arquitetura, dados, treino e produto.
- **É fácil de expandir**, permitindo testar novos schedulers, losses e estratégias de refinamento.
- **Tem valor educacional alto**, porque o código é relativamente modular e estudável.
- **Serve como base para novos experimentos**, incluindo benchmarks, quantização, serving e melhorias de qualidade.

---

## Casos de uso ideais

O DiffuGPT é mais adequado para:
- experimentação com modelos híbridos;
- estudo de geração de texto por difusão discreta;
- pesquisa em decodificação híbrida;
- protótipos locais de inferência;
- projetos educacionais e de portfólio.

Ele **ainda não é** um foundation model pronto para produção comercial em larga escala.

---

## Roadmap

- métricas e benchmarks mais fortes;
- datasets maiores e mais limpos;
- scheduler e objetivo diffusion mais robustos;
- relatórios de treino e avaliação mais completos;
- API e deploy mais maduros;
- model card e benchmark card;
- otimização de inferência e quantização.

---

## Limpeza de arquivos gerados

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

## Licença

MIT


## O que melhorou

- Dei mais força ao topo do README com proposta de valor clara, badges enxutos e uma lista de highlights, porque badges relevantes no começo e em quantidade moderada tendem a melhorar legibilidade e credibilidade.[1]
- Estruturei o texto como README de projeto de ML com toque de model card leve, incluindo seções como `Why this project is interesting`, `Intended use` e `Roadmap`, porque essa documentação ajuda a explicar utilidade, limites e descoberta do modelo.[2][3][5]
- Mantive tabelas e blocos de arquitetura porque o GitHub renderiza bem esse formato e ele funciona melhor para parâmetros, modos e estrutura do repositório.[6][7]
- Evitei excesso de badges e textos repetidos, porque a própria orientação comum para READMEs é priorizar clareza, relevância e posicionamento estratégico do que importa no topo.[4][1]

## Pontos positivos para destacar

- Projeto construído do zero, incluindo tokenizer, modelo, treino e inferência.
- Arquitetura híbrida incomum e interessante para pesquisa.
- Código modular e relativamente fácil de estudar.
- Pipeline completo: dados, tokenizer, treino, CLI e web.
- Bom apelo para portfólio técnico e conteúdo de demonstração, especialmente porque você quer apresentá-lo de forma profissional.

Se quiser, eu posso fazer a próxima etapa e te devolver isso já em uma versão **README final de GitHub**, com emojis mais equilibrados, badges extras e seção de benchmark visual.
