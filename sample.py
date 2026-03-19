# sample.py

import torch
import time
from config import Config
from model.hybrid_model import HybridModel
from tokenizer import BPEFastTokenizer


# ══════════════════════════════════════════════════════════════════
# MODO 1 — Diffusion  (bidirecional, iterativa)
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

        probs      = torch.softmax(logits / 0.5, dim=-1)
        confidence = probs.max(dim=-1).values

        current_tokens = (x != model.MASK).sum().item()
        generated      = current_tokens - initial_tokens

        if generated < min_new_tokens:
            update_mask = (x == model.MASK)
        else:
            update_mask = (x == model.MASK) | (confidence < 0.6)

        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(x.shape)

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
    x   = torch.tensor(ids).unsqueeze(0).to(device)

    start     = time.time()
    generated = 0

    while generated < max_new_tokens and x.shape[1] < cfg.block_size:
        with torch.no_grad():
            logits, _ = model.forward_ar(x)

        next_logits = logits[0, -1] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = float("-inf")

        probs      = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        x         = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        generated += 1

    elapsed = time.time() - start
    tps     = generated / max(elapsed, 1e-6)

    return tok.decode(x[0].tolist()), generated, tps


# ══════════════════════════════════════════════════════════════════
# MODO 3 — Hibrido  (Diffusion gera rascunho → AR corrige)
# ══════════════════════════════════════════════════════════════════
def _refine_ar(model, draft_ids, cfg, temperature=1.0, threshold=0.65):
    """
    Versão rápida: UM único forward pass causal no rascunho inteiro.
    logits[i] prediz o token na posição i+1 — corrige se confiante.
    """
    device = cfg.device
    x      = torch.tensor(draft_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model.forward_ar(x)          # (1, T, V) — um único pass

    probs              = torch.softmax(logits[0] / temperature, dim=-1)  # (T, V)
    top_probs, top_tok = probs.max(dim=-1)        # (T,)

    refined = list(draft_ids)
    for i in range(len(draft_ids) - 1):
        # logits na posição i prediz o token i+1
        if top_probs[i].item() > threshold and top_tok[i].item() != refined[i + 1]:
            refined[i + 1] = top_tok[i].item()

    return refined


def generate_hybrid(
    model,
    tok,
    prompt,
    diff_steps=24,
    min_new_tokens=20,
    ar_temperature=1.0,
    ar_threshold=0.65,
    show_steps=False,
):
    """
    Pipeline hibrido em 2 estagios:
      1) Diffusion  → gera rascunho global
      2) AR Refiner → corrige token por token
    """
    cfg = Config()

    # Estagio 1: Diffusion
    draft_text, diff_tokens, diff_tps = generate(
        model, tok, prompt,
        steps=diff_steps,
        min_new_tokens=min_new_tokens,
        show_steps=show_steps,
        inline=False,
    )

    # Estagio 2: AR Refinement
    start     = time.time()
    draft_ids = tok.encode(draft_text)
    refined   = _refine_ar(model, draft_ids, cfg, ar_temperature, ar_threshold)
    elapsed   = time.time() - start

    refined_text = tok.decode(refined)
    ar_tps       = len(refined) / max(elapsed, 1e-6)

    return draft_text, refined_text, diff_tokens, diff_tps, len(refined), ar_tps


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
    print("Modos disponiveis:")
    print("  1) Diffusion pura")
    print("  2) AR puro (GPT-style)")
    print("  3) Hibrido (Diffusion + AR refiner)\n")

    modo   = input("Modo (1/2/3): ").strip()
    prompt = input("Prompt: ").strip()

    if modo == "1":
        show = input("Mostrar difusao? (s/n): ").lower() == "s"
        print("\nGerando...\n")
        out, tokens, tps = generate(model, tok, prompt, show_steps=show, inline=True)
        print(f"\n\n=== RESULTADO FINAL ===\n{out}")
        print(f"\nTokens: {tokens} | Velocidade: {tps:.2f} tok/s")

    elif modo == "2":
        temp  = float(input("Temperatura (ex: 1.0): ") or "1.0")
        n_tok = int(input("Max novos tokens (ex: 80): ") or "80")
        print("\nGerando...\n")
        out, tokens, tps = generate_ar(model, tok, prompt, max_new_tokens=n_tok, temperature=temp)
        print(f"=== RESULTADO FINAL ===\n{out}")
        print(f"\nTokens: {tokens} | Velocidade: {tps:.2f} tok/s")

    elif modo == "3":
        show = input("Mostrar difusao? (s/n): ").lower() == "s"
        print("\nGerando...\n")
        draft, refined, dt, dtps, at, atps = generate_hybrid(
            model, tok, prompt, show_steps=show
        )
        print(f"\n\n=== RASCUNHO (difusao) ===\n{draft}")
        print(f"\n=== RESULTADO FINAL (apos AR) ===\n{refined}")
        print(f"\n=== METRICAS ===")
        print(f"Difusao : {dt} tokens @ {dtps:.2f} tok/s")
        print(f"AR      : {at} tokens @ {atps:.2f} tok/s")
    else:
        print("Modo invalido.")


if __name__ == "__main__":
    main()
