# model/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridModel(nn.Module):
    """
    Modelo híbrido Diffusion + AR.

    - forward_diffusion(): atenção bidirecional, prevê tokens mascarados.
    - forward_ar()       : atenção causal (GPT), prevê próximo token.

    Embeddings de token, posição e lm_head são COMPARTILHADOS.
    O time_emb só é somado no modo diffusion.
    """

    def __init__(self, vocab_size, block_size, emb_dim, n_layers, n_heads, max_timesteps):
        super().__init__()

        self.vocab_size    = vocab_size
        self.block_size    = block_size
        self.MASK          = vocab_size - 1
        self.max_timesteps = max_timesteps

        # ── embeddings compartilhados ──────────────────────────────
        self.tok_emb  = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb  = nn.Embedding(block_size,  emb_dim)
        self.time_emb = nn.Embedding(max_timesteps, emb_dim)  # só diffusion

        # ── transformer compartilhado ──────────────────────────────
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # ── cabeça de predição compartilhada ──────────────────────
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    # ── utilidade: máscara causal (triangular superior) ───────────
    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    # ══════════════════════════════════════════════════════════════
    # MODO 1 — Diffusion  (bidirecional, sem máscara causal)
    # ══════════════════════════════════════════════════════════════
    def forward_diffusion(self, x_ids, t, targets=None):
        """
        x_ids  : (B, T) sequência parcialmente mascarada
        t      : (B,)   timestep do ruído
        targets: (B, T) sequência original (para calcular loss)
        """
        B, T = x_ids.shape

        tok  = self.tok_emb(x_ids)                                    # (B, T, D)
        pos  = self.pos_emb(torch.arange(T, device=x_ids.device))     # (T, D)
        time = self.time_emb(t)[:, None, :]                           # (B, 1, D)

        x = tok + pos[None] + time           # timestep somado a todos os tokens
        x = self.transformer(x)              # sem máscara → bidirecional
        logits = self.lm_head(x)             # (B, T, V)

        loss = None
        if targets is not None:
            mask = (x_ids == self.MASK)
            if mask.any():
                loss = F.cross_entropy(logits[mask], targets[mask])
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                )
        return logits, loss

    # ══════════════════════════════════════════════════════════════
    # MODO 2 — AR Refiner  (causal, token por token, estilo GPT)
    # ══════════════════════════════════════════════════════════════
    def forward_ar(self, x_ids, targets=None):
        """
        x_ids  : (B, T) sequência de entrada
        targets: (B, T) sequência deslocada +1 (para calcular loss)

        Usa máscara causal → cada posição só vê o que vem antes dela.
        """
        B, T = x_ids.shape

        tok    = self.tok_emb(x_ids)
        pos    = self.pos_emb(torch.arange(T, device=x_ids.device))
        causal = self._causal_mask(T, x_ids.device)

        x      = tok + pos[None]             # sem time_emb no modo AR
        x      = self.transformer(x, mask=causal)
        logits = self.lm_head(x)             # (B, T, V)

        loss = None
        if targets is not None:
            # GPT-style: prever token i+1 a partir de token i
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.vocab_size),
                targets[:, 1:].contiguous().view(-1),
            )
        return logits, loss
