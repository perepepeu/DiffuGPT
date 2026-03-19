# sample.py

import torch
import time
from config import Config
from model.hybrid_model import HybridModel
from tokenizer import BPEFastTokenizer


# ══════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════
def _sanitize_ids(ids, vocab_size):
    """Garante que nenhum token id esta fora do range valido."""
    return [max(0, min(i, vocab_size - 2)) for i in ids]


def _safe_probs(logits, temperature=0.5):
    """
    Converte logits em probabilidades de forma segura:
    1. Remove NaN e Inf dos logits
    2. Aplica softmax
    3. Remove NaN das probs (pode surgir se todos logits forem -inf)
    4. Renormaliza
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    probs  = torch.softmax(logits / temperature, dim=-1)
    probs  = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    # renormaliza linha por linha para garantir soma = 1
    s = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs / s


# ══════════════════════════════════════════════════════════════════
# MODO 1 — Diffusion
# ══════════════════════════════════════════════════════════════════
def generate(
    model,
    tok,
    prompt,
    steps=24,
    min_new_tokens=20,
    show_steps=False,
    inline=True,
):
    cfg = Config()

    x   = torch.full((1, cfg.block_size), model.MASK).long().to(cfg.device)
    ids = tok.encode(prompt)
    x[0, : len(ids)] = torch.tensor(ids).to(cfg.device)

    start          = time.time()
    max_t          = model.max_timesteps
    initial_tokens = len(ids)

    for t in reversed(range(1, max_t)):
        t_tensor = torch.tensor([t]).to(cfg.device)

        with torch.no_grad():
            logits, _ = model.forward_diffusion(x, t_tensor)

        # ← FIX: safe probs (remove NaN/Inf + renormaliza)
        probs      = _safe_probs(logits, temperature=0.5)
        confidence = probs.max(dim=-1).values

        current_tokens = (x != model.MASK).sum().item()
        generated      = current_tokens - initial_tokens

        if generated < min_new_tokens:
            update_mask = (x == model.MASK)
        else:
            update_mask = (x == model.MASK) | (confidence < 0.6)

        # ← FIX: probs já seguras, clamp no resultado do multinomial
        probs_flat = probs.view(-1, probs.size(-1))
        sampled    = torch.multinomial(probs_flat, 1).view(x.shape)
        sampled    = sampled.clamp(0, cfg.vocab_size - 2)

        x[update_mask] = sampled[update_mask]

        if show_steps:
            decoded = tok.decode(x[0].tolist())
            if inline:
                print(f"\r[t={t}] {decoded[:120]}", end="")
            else:
                print(f"\n[t={t}] {decoded}")

    end = time.time()

    total_tokens     = (x != model.MASK).sum().item()
    generated_tokens = total_tokens - initial_tokens
    elapsed          = end - start

    return tok.decode(x[0].tolist()), generated_tokens, generated_tokens / max(elapsed, 1e-6)


# ══════════════════════════════════════════════════════════════════
# MODO 2 — AR puro  (GPT-style, token por token)
# ══════════════════════════════════════════════════════════════════
def generate_ar(
    model,
    tok,
    prompt,
    max_new_tokens=80,
    temperature=1.0,
    top_k=40,
):
    cfg    = Config()
    device = cfg.device

    ids = tok.encode(prompt)
    ids = _sanitize_ids(ids, cfg.vocab_size)
    x   = torch.tensor(ids).unsqueeze(0).to(device)

    start     = time.time()
    generated = 0

    while generated < max_new_tokens and x.shape[1] < cfg.block_size:
        with torch.no_grad():
            logits, _ = model.forward_ar(x)

        next_logits = logits[0, -1]
        # ← FIX: remove NaN/Inf antes de qualquer operacao
        next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        next_logits = next_logits / temperature

        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = float("-inf")

        probs      = torch.softmax(next_logits, dim=-1)
        probs      = torch.nan_to_num(probs, nan=0.0)
        probs      = probs / probs.sum().clamp(min=1e-8)

        next_token = torch.multinomial(probs, 1)
        next_token = next_token.clamp(0, cfg.vocab_size - 2)

        x         = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        generated += 1

    elapsed = time.time() - start
    return tok.decode(x[0].tolist()), generated, generated / max(elapsed, 1e-6)


# ══════════════════════════════════════════════════════════════════
# REFINADOR AR — batched (1 forward pass, sempre rapido)
# ══════════════════════════════════════════════════════════════════
def _refine_ar(model, draft_ids, cfg, temperature=1.0, mode="balanced"):
    threshold = cfg.ar_threshold_fast if mode == "fast" else cfg.ar_threshold_balanced
    device    = cfg.device

    draft_ids = _sanitize_ids(draft_ids, cfg.vocab_size)
    x         = torch.tensor(draft_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model.forward_ar(x)

    # ← FIX: safe probs no refinador tambem
    probs              = _safe_probs(logits[0], temperature=temperature)
    top_probs, top_tok = probs.max(dim=-1)

    refined  = list(draft_ids)
    replaced = 0
    for i in range(len(draft_ids) - 1):
        pred = top_tok[i].item()
        pred = max(0, min(pred, cfg.vocab_size - 2))
        if top_probs[i].item() > threshold and pred != refined[i + 1]:
            refined[i + 1] = pred
            replaced += 1

    return refined, replaced


# ══════════════════════════════════════════════════════════════════
# MODO 3 — Hibrido  (Diffusion + AR refiner)
# ══════════════════════════════════════════════════════════════════
def generate_hybrid(
    model,
    tok,
    prompt,
    diff_steps=24,
    min_new_tokens=20,
    ar_temperature=1.0,
    ar_mode=None,
    show_steps=False,
):
    cfg      = Config()
    refine_m = ar_mode if ar_mode else cfg.ar_refine_mode

    draft_text, diff_tokens, diff_tps = generate(
        model, tok, prompt,
        steps=diff_steps,
        min_new_tokens=min_new_tokens,
        show_steps=show_steps,
        inline=False,
    )

    start     = time.time()
    draft_ids = tok.encode(draft_text)
    draft_ids = _sanitize_ids(draft_ids, cfg.vocab_size)

    refined, replaced = _refine_ar(model, draft_ids, cfg, ar_temperature, mode=refine_m)
    elapsed           = time.time() - start

    refined_text = tok.decode(refined)
    ar_tps       = len(refined) / max(elapsed, 1e-6)

    return draft_text, refined_text, diff_tokens, diff_tps, len(refined), ar_tps, replaced


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main():
    cfg = Config()

    tok = BPEFastTokenizer()
    tok.load("data/tok-1024.model")

    model = HybridModel(
        cfg.vocab_size, cfg.block_size, cfg.emb_dim,
        cfg.n_layers,   cfg.n_heads,    cfg.max_timesteps,
    ).to(cfg.device)

    sd = torch.load("model.pt", map_location=cfg.device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    print("=== HYBRID TEXT GENERATOR ===\n")
    print("  1) Diffusion pura")
    print("  2) AR puro (GPT-style)")
    print("  3) Hibrido — balanced (meio-termo, padrao)")
    print("  4) Hibrido — fast    (velocidade maxima)\n")

    modo   = input("Modo (1/2/3/4): ").strip()
    prompt = input("Prompt: ").strip()

    if modo == "1":
        show = input("Mostrar difusao? (s/n): ").lower() == "s"
        print("\nGerando...\n")
        out, tokens, tps = generate(model, tok, prompt, show_steps=show, inline=True)
        print(f"\n\n=== RESULTADO ===\n{out}")
        print(f"Tokens: {tokens} | {tps:.2f} tok/s")

    elif modo == "2":
        temp  = float(input("Temperatura (ex: 1.0): ") or "1.0")
        n_tok = int(input("Max novos tokens (ex: 80): ") or "80")
        print("\nGerando...\n")
        out, tokens, tps = generate_ar(model, tok, prompt, max_new_tokens=n_tok, temperature=temp)
        print(f"=== RESULTADO ===\n{out}")
        print(f"Tokens: {tokens} | {tps:.2f} tok/s")

    elif modo in ("3", "4"):
        refine_mode = "balanced" if modo == "3" else "fast"
        show = input("Mostrar difusao? (s/n): ").lower() == "s"
        print("\nGerando...\n")
        draft, refined, dt, dtps, at, atps, replaced = generate_hybrid(
            model, tok, prompt, show_steps=show, ar_mode=refine_mode
        )
        print(f"\n\n=== RASCUNHO (difusao) ===\n{draft}")
        print(f"\n=== RESULTADO FINAL (apos AR [{refine_mode}]) ===\n{refined}")
        print(f"\n=== METRICAS ===")
        print(f"Difusao : {dt} tokens @ {dtps:.2f} tok/s")
        print(f"AR      : {at} tokens @ {atps:.2f} tok/s  ({replaced} tokens corrigidos)")
    else:
        print("Modo invalido.")


if __name__ == "__main__":
    main()