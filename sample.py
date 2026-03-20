# sample.py
import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from config import Config
from model.hybrid_model import HybridModel
from tokenizer import BPEFastTokenizer


# Estrutura padronizada para retornar resultados de geração simples.
@dataclass
class GenerationResult:
    text: str
    ids: List[int]
    prompt_tokens: int
    generated_tokens: int
    tokens_per_sec: float


# Estrutura padronizada para retornar resultados do modo híbrido
# (rascunho por difusão + refinamento autoregressivo).
@dataclass
class HybridResult:
    draft: GenerationResult
    refined_text: str
    refined_ids: List[int]
    refined_tokens_processed: int
    refined_tokens_replaced: int
    refine_tokens_per_sec: float


# Lê um atributo do config com fallback para valor padrão.
def get_cfg_value(cfg, name, default):
    return getattr(cfg, name, default)


# Carrega checkpoint de forma compatível com versões diferentes do PyTorch.
def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# Remove prefixos adicionados por compilação/empacotamento do modelo
# e extrai o state_dict correto quando o checkpoint vem aninhado.
def strip_compiled_prefix(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint inválido: esperado dict com pesos do modelo.")

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


# Carrega o tokenizer e valida se ele possui um vocab_size utilizável.
def load_tokenizer(tokenizer_path: str) -> BPEFastTokenizer:
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer não encontrado: {tokenizer_path}")

    tok = BPEFastTokenizer()
    tok.load(tokenizer_path)

    if not hasattr(tok, "vocab_size"):
        raise AttributeError("O tokenizer carregado não possui atributo vocab_size.")

    if not isinstance(tok.vocab_size, int) or tok.vocab_size <= 0:
        raise ValueError(f"vocab_size inválido no tokenizer: {tok.vocab_size}")

    return tok


# Instancia o modelo com base nos hiperparâmetros presentes em cfg.
def build_model(cfg: Config, device: torch.device) -> HybridModel:
    model = HybridModel(
        cfg.vocab_size,
        cfg.block_size,
        cfg.emb_dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.max_timesteps,
    ).to(device)
    return model


# Carrega o modelo e injeta os pesos do checkpoint, validando compatibilidade
# entre arquitetura, tokenizer e arquivo model.pt.
def load_model(cfg: Config, model_path: str) -> HybridModel:
    device = torch.device(cfg.device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")

    model = build_model(cfg, device)
    checkpoint = safe_torch_load(model_path, map_location=device)
    state_dict = strip_compiled_prefix(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Falha ao carregar model.pt. Verifique se config.py, tokenizer e checkpoint "
            "foram gerados com a mesma arquitetura."
        ) from e

    model.eval()
    return model


# Garante que todos os token IDs estejam dentro do vocabulário esperado.
def ensure_ids_in_range(
    ids: List[int], vocab_size: int, source: str = "ids"
) -> List[int]:
    if not isinstance(ids, list):
        ids = list(ids)

    cleaned = []
    for i, token_id in enumerate(ids):
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"{source}: token id fora do range na posição {i}: {token_id} "
                f"(vocab_size={vocab_size})"
            )
        cleaned.append(token_id)

    return cleaned


# Tokeniza o prompt e valida se os IDs gerados são válidos.
def encode_prompt(tok: BPEFastTokenizer, prompt: str, vocab_size: int) -> List[int]:
    ids = tok.encode(prompt)
    ids = ensure_ids_in_range(ids, vocab_size, source="prompt")
    return ids


# Remove máscaras no final da sequência para evitar lixo na decodificação.
def trim_trailing_mask(ids: List[int], mask_token: int) -> List[int]:
    end = len(ids)
    while end > 0 and ids[end - 1] == mask_token:
        end -= 1
    return ids[:end]


# Converte IDs em texto garantindo tipo inteiro na entrada do tokenizer.
def decode_ids(tok: BPEFastTokenizer, ids: List[int]) -> str:
    return tok.decode([int(i) for i in ids])


# Softmax robusto contra NaN/Inf e temperatura muito baixa.
def safe_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(logits / temperature, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs / denom


# Restringe a amostragem aos k logits mais prováveis.
def apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None:
        return logits

    top_k = int(top_k)
    if top_k <= 0:
        return logits

    k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    filtered = logits.clone()
    filtered[filtered < threshold] = float("-inf")
    return filtered


# Garante que o número de passos de difusão caiba no limite temporal do modelo.
def clamp_steps(steps: int, max_timesteps: int) -> int:
    if max_timesteps < 2:
        raise ValueError("model.max_timesteps precisa ser >= 2.")

    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps precisa ser >= 1.")

    return min(steps, max_timesteps - 1)


# Se o prompt for maior que a janela de contexto, mantém apenas a parte final.
def prepare_prompt_ids(ids: List[int], block_size: int) -> Tuple[List[int], bool]:
    truncated = len(ids) > block_size
    if truncated:
        ids = ids[-block_size:]
    return ids, truncated


# Mostra uma prévia compacta do texto gerado durante a difusão.
def print_step_preview(prefix: str, text: str, inline: bool):
    text = text.replace("\n", "\\n")
    preview = text[:160]
    if inline:
        print(f"\r{prefix} {preview}", end="", flush=True)
    else:
        print(f"{prefix} {preview}")


# Geração por difusão: começa com máscaras e vai refinando os tokens ao longo
# dos timesteps até formar a sequência final.
@torch.inference_mode()
def generate_diffusion(
    model: HybridModel,
    tok: BPEFastTokenizer,
    cfg: Config,
    prompt: str,
    steps: int = 24,
    min_new_tokens: int = 20,
    temperature: float = 0.5,
    confidence_threshold: float = 0.6,
    show_steps: bool = False,
    inline: bool = True,
) -> GenerationResult:
    device = torch.device(cfg.device)
    prompt_ids = encode_prompt(tok, prompt, cfg.vocab_size)
    prompt_ids, _ = prepare_prompt_ids(prompt_ids, cfg.block_size)

    if not (0 <= model.MASK < cfg.vocab_size):
        raise ValueError("model.MASK fora do range do vocabulário.")

    prompt_len = len(prompt_ids)
    editable_len = cfg.block_size - prompt_len

    # Inicializa toda a sequência com MASK e preserva o prompt no início.
    x = torch.full(
        (1, cfg.block_size),
        model.MASK,
        dtype=torch.long,
        device=device,
    )

    if prompt_len > 0:
        x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # Se o prompt já ocupou toda a janela, apenas devolve o texto atual.
    if editable_len <= 0:
        final_ids = x[0].tolist()
        final_ids = trim_trailing_mask(final_ids, model.MASK)
        return GenerationResult(
            text=decode_ids(tok, final_ids),
            ids=final_ids,
            prompt_tokens=prompt_len,
            generated_tokens=0,
            tokens_per_sec=0.0,
        )

    editable_mask = torch.zeros_like(x, dtype=torch.bool)
    editable_mask[:, prompt_len:] = True

    steps = clamp_steps(steps, model.max_timesteps)
    start_t = steps
    start = time.time()

    for t in range(start_t, 0, -1):
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)
        logits, _ = model.forward_diffusion(x, t_tensor)

        probs = safe_softmax(logits, temperature=temperature)
        confidence = probs.max(dim=-1).values

        # Antes de atingir o mínimo desejado, só preenche posições mascaradas.
        # Depois disso, também permite revisar tokens de baixa confiança.
        generated_so_far = int((x[:, prompt_len:] != model.MASK).sum().item())
        if generated_so_far < min_new_tokens:
            update_mask = editable_mask & (x == model.MASK)
        else:
            update_mask = editable_mask & (
                (x == model.MASK) | (confidence < confidence_threshold)
            )

        if update_mask.any():
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(x)
            x[update_mask] = sampled[update_mask]

        if show_steps:
            preview_ids = x[0].tolist()
            preview_ids = trim_trailing_mask(preview_ids, model.MASK)
            preview_text = decode_ids(tok, preview_ids)
            print_step_preview(f"[t={t:02d}]", preview_text, inline=inline)

    # Preenche máscaras remanescentes com argmax no último passo.
    remaining_mask = editable_mask & (x == model.MASK)
    if remaining_mask.any():
        t_tensor = torch.tensor([1], dtype=torch.long, device=device)
        logits, _ = model.forward_diffusion(x, t_tensor)
        fill_ids = torch.argmax(logits, dim=-1)
        x[remaining_mask] = fill_ids[remaining_mask]

    elapsed = time.time() - start

    final_ids = x[0].tolist()
    final_ids = trim_trailing_mask(final_ids, model.MASK)
    final_text = decode_ids(tok, final_ids)

    generated_tokens = max(0, len(final_ids) - prompt_len)
    tps = generated_tokens / max(elapsed, 1e-6)

    if show_steps and inline:
        print()

    return GenerationResult(
        text=final_text,
        ids=final_ids,
        prompt_tokens=prompt_len,
        generated_tokens=generated_tokens,
        tokens_per_sec=tps,
    )


# Geração autoregressiva clássica: prevê um token por vez, anexando ao contexto.
@torch.inference_mode()
def generate_ar(
    model: HybridModel,
    tok: BPEFastTokenizer,
    cfg: Config,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    top_k: Optional[int] = 40,
) -> GenerationResult:
    device = torch.device(cfg.device)
    prompt_ids = encode_prompt(tok, prompt, cfg.vocab_size)

    if len(prompt_ids) == 0:
        raise ValueError(
            "Prompt vazio. O modo AR precisa de pelo menos 1 token de entrada."
        )

    prompt_ids, _ = prepare_prompt_ids(prompt_ids, cfg.block_size)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    prompt_len = x.size(1)
    if prompt_len >= cfg.block_size or max_new_tokens <= 0:
        final_ids = x[0].tolist()
        return GenerationResult(
            text=decode_ids(tok, final_ids),
            ids=final_ids,
            prompt_tokens=prompt_len,
            generated_tokens=0,
            tokens_per_sec=0.0,
        )

    start = time.time()
    generated = 0

    while generated < max_new_tokens and x.size(1) < cfg.block_size:
        logits, _ = model.forward_ar(x)

        next_logits = logits[0, -1]
        next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        next_logits = next_logits / max(float(temperature), 1e-6)
        next_logits = apply_top_k(next_logits, top_k)

        # Evita distribuição inválida se todos os logits forem filtrados.
        if torch.isneginf(next_logits).all():
            next_logits = torch.zeros_like(next_logits)

        probs = torch.softmax(next_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum().clamp(min=1e-8)

        next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
        x = torch.cat([x, next_token], dim=1)
        generated += 1

    elapsed = time.time() - start
    final_ids = ensure_ids_in_range(x[0].tolist(), cfg.vocab_size, source="saída AR")

    return GenerationResult(
        text=decode_ids(tok, final_ids),
        ids=final_ids,
        prompt_tokens=prompt_len,
        generated_tokens=generated,
        tokens_per_sec=generated / max(elapsed, 1e-6),
    )


# Repassa o texto gerado token a token usando o modo AR para corrigir posições
# nas quais o modelo estiver suficientemente confiante em outra previsão.
@torch.inference_mode()
def refine_with_ar(
    model: HybridModel,
    draft_ids: List[int],
    cfg: Config,
    protected_prefix_len: int,
    temperature: float = 1.0,
    mode: str = "balanced",
) -> Tuple[List[int], int, int]:
    device = torch.device(cfg.device)
    draft_ids = ensure_ids_in_range(draft_ids, cfg.vocab_size, source="draft_ids")

    threshold_fast = float(get_cfg_value(cfg, "ar_threshold_fast", 0.90))
    threshold_balanced = float(get_cfg_value(cfg, "ar_threshold_balanced", 0.70))
    threshold = threshold_fast if mode == "fast" else threshold_balanced

    refined = list(draft_ids)
    protected_prefix_len = max(0, min(int(protected_prefix_len), len(refined)))

    replaced = 0
    processed = 0

    for target_idx in range(max(1, protected_prefix_len), len(refined)):
        prefix = refined[:target_idx]
        if len(prefix) > cfg.block_size:
            prefix = prefix[-cfg.block_size :]

        x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model.forward_ar(x)

        next_logits = logits[0, -1]
        probs = safe_softmax(next_logits, temperature=temperature)

        top_prob, top_tok = probs.max(dim=-1)
        pred = int(top_tok.item())

        processed += 1
        if float(top_prob.item()) > threshold and pred != refined[target_idx]:
            refined[target_idx] = pred
            replaced += 1

    return refined, replaced, processed


# Pipeline híbrido: primeiro gera um rascunho por difusão e depois usa
# o modo AR para revisar parte da sequência.
def generate_hybrid(
    model: HybridModel,
    tok: BPEFastTokenizer,
    cfg: Config,
    prompt: str,
    diff_steps: int = 24,
    min_new_tokens: int = 20,
    diff_temperature: float = 0.5,
    diff_confidence_threshold: float = 0.6,
    ar_temperature: float = 1.0,
    ar_mode: Optional[str] = None,
    show_steps: bool = False,
) -> HybridResult:
    refine_mode = ar_mode or get_cfg_value(cfg, "ar_refine_mode", "balanced")

    draft = generate_diffusion(
        model=model,
        tok=tok,
        cfg=cfg,
        prompt=prompt,
        steps=diff_steps,
        min_new_tokens=min_new_tokens,
        temperature=diff_temperature,
        confidence_threshold=diff_confidence_threshold,
        show_steps=show_steps,
        inline=False,
    )

    start = time.time()
    refined_ids, replaced, processed = refine_with_ar(
        model=model,
        draft_ids=draft.ids,
        cfg=cfg,
        protected_prefix_len=draft.prompt_tokens,
        temperature=ar_temperature,
        mode=refine_mode,
    )
    elapsed = time.time() - start

    return HybridResult(
        draft=draft,
        refined_text=decode_ids(tok, refined_ids),
        refined_ids=refined_ids,
        refined_tokens_processed=processed,
        refined_tokens_replaced=replaced,
        refine_tokens_per_sec=processed / max(elapsed, 1e-6),
    )


# Normaliza aliases de entrada para os nomes internos dos modos.
def normalize_mode(raw: str) -> str:
    raw = (raw or "").strip().lower()

    mapping = {
        "1": "diffusion",
        "2": "ar",
        "3": "hybrid-balanced",
        "4": "hybrid-fast",
        "diffusion": "diffusion",
        "ar": "ar",
        "hybrid": "hybrid-balanced",
        "hybrid-balanced": "hybrid-balanced",
        "hybrid-fast": "hybrid-fast",
        "balanced": "hybrid-balanced",
        "fast": "hybrid-fast",
    }

    if raw not in mapping:
        raise ValueError(
            "Modo inválido. Use 1, 2, 3, 4, diffusion, ar, hybrid-balanced ou hybrid-fast."
        )

    return mapping[raw]


# Lê uma string do terminal com suporte a valor padrão.
def ask_str(prompt: str, default: Optional[str] = None) -> str:
    value = input(prompt).strip()
    if value:
        return value
    if default is not None:
        return default
    return ""


# Lê um inteiro do terminal; em caso de erro, mantém o padrão.
def ask_int(prompt: str, default: int) -> int:
    value = input(prompt).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Lê um float do terminal; aceita vírgula como separador decimal.
def ask_float(prompt: str, default: float) -> float:
    value = input(prompt).strip().replace(",", ".")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Define os argumentos aceitos pela interface de linha de comando.
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Hybrid text generator (Diffusion + AR)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="1/2/3/4 ou diffusion/ar/hybrid-balanced/hybrid-fast",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt de entrada")
    parser.add_argument(
        "--model-path", type=str, default="model.pt", help="Caminho do checkpoint"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="data/tok-1024.model",
        help="Caminho do tokenizer",
    )
    parser.add_argument("--steps", type=int, default=24, help="Passos de difusão")
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=20,
        help="Mínimo de tokens novos na difusão",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperatura AR ou refinador"
    )
    parser.add_argument(
        "--diff-temperature", type=float, default=0.5, help="Temperatura da difusão"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Threshold de confiança da difusão",
    )
    parser.add_argument("--top-k", type=int, default=40, help="Top-k do modo AR")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Máximo de novos tokens no modo AR",
    )
    parser.add_argument(
        "--show-steps", action="store_true", help="Mostra progresso da difusão"
    )
    return parser


# Coleta parâmetros interativamente quando prompt ou modo não vierem pela CLI.
def interactive_config(args):
    print("=== HYBRID TEXT GENERATOR ===\n")
    print("  1) Diffusion pura")
    print("  2) AR puro (GPT-style)")
    print("  3) Híbrido — balanced")
    print("  4) Híbrido — fast\n")

    mode = normalize_mode(args.mode or ask_str("Modo (1/2/3/4): "))
    prompt = args.prompt if args.prompt is not None else ask_str("Prompt: ")

    if not prompt.strip():
        raise ValueError("Prompt vazio.")

    if mode == "diffusion":
        show_steps = (
            args.show_steps or ask_str("Mostrar difusão? (s/n): ", "n").lower() == "s"
        )
        args.mode = mode
        args.prompt = prompt
        args.show_steps = show_steps
        return args

    if mode == "ar":
        args.temperature = ask_float(
            f"Temperatura [{args.temperature}]: ", args.temperature
        )
        args.max_new_tokens = ask_int(
            f"Max novos tokens [{args.max_new_tokens}]: ", args.max_new_tokens
        )
        args.mode = mode
        args.prompt = prompt
        return args

    show_steps = (
        args.show_steps or ask_str("Mostrar difusão? (s/n): ", "n").lower() == "s"
    )
    args.mode = mode
    args.prompt = prompt
    args.show_steps = show_steps
    args.temperature = ask_float(
        f"Temperatura AR [{args.temperature}]: ", args.temperature
    )
    return args


# Ponto de entrada principal: carrega config, tokenizer e modelo;
# escolhe o modo; executa a geração; e imprime métricas.
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        cfg = Config()
        tok = load_tokenizer(args.tokenizer_path)

        # Sincroniza o vocab_size do config com o tokenizer carregado.
        if hasattr(tok, "vocab_size"):
            cfg.vocab_size = tok.vocab_size

        model = load_model(cfg, args.model_path)

        # Se faltarem argumentos obrigatórios, entra no modo interativo.
        if args.mode is None or args.prompt is None:
            args = interactive_config(args)
        else:
            args.mode = normalize_mode(args.mode)
            if not args.prompt.strip():
                raise ValueError("Prompt vazio.")

        print("\nGerando...\n")

        if args.mode == "diffusion":
            result = generate_diffusion(
                model=model,
                tok=tok,
                cfg=cfg,
                prompt=args.prompt,
                steps=args.steps,
                min_new_tokens=args.min_new_tokens,
                temperature=args.diff_temperature,
                confidence_threshold=args.confidence_threshold,
                show_steps=args.show_steps,
                inline=True,
            )
            print(f"\n=== RESULTADO ===\n{result.text}")
            print(
                f"Tokens gerados: {result.generated_tokens} | {result.tokens_per_sec:.2f} tok/s"
            )
            return

        if args.mode == "ar":
            result = generate_ar(
                model=model,
                tok=tok,
                cfg=cfg,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"=== RESULTADO ===\n{result.text}")
            print(
                f"Tokens gerados: {result.generated_tokens} | {result.tokens_per_sec:.2f} tok/s"
            )
            return

        ar_mode = "balanced" if args.mode == "hybrid-balanced" else "fast"
        result = generate_hybrid(
            model=model,
            tok=tok,
            cfg=cfg,
            prompt=args.prompt,
            diff_steps=args.steps,
            min_new_tokens=args.min_new_tokens,
            diff_temperature=args.diff_temperature,
            diff_confidence_threshold=args.confidence_threshold,
            ar_temperature=args.temperature,
            ar_mode=ar_mode,
            show_steps=args.show_steps,
        )

        print(f"\n=== RASCUNHO (difusão) ===\n{result.draft.text}")
        print(f"\n=== RESULTADO FINAL (após AR [{ar_mode}]) ===\n{result.refined_text}")
        print("\n=== MÉTRICAS ===")
        print(
            f"Difusão : {result.draft.generated_tokens} tokens @ {result.draft.tokens_per_sec:.2f} tok/s"
        )
        print(
            f"AR      : {result.refined_tokens_processed} tokens processados @ "
            f"{result.refine_tokens_per_sec:.2f} tok/s "
            f"({result.refined_tokens_replaced} tokens corrigidos)"
        )

    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")
    except Exception as e:
        print(f"\nErro: {e}")


if __name__ == "__main__":
    main()
