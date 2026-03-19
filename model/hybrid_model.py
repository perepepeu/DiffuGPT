# model/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridModel(nn.Module):
    """
    Modelo híbrido Diffusion + AR.

    Modos:
    - forward_diffusion(): atenção bidirecional para reconstrução de tokens mascarados.
    - forward_ar(): atenção causal para previsão autoregressiva estilo GPT.

    Observações:
    - MASK continua sendo vocab_size - 1; esse id precisa estar reservado no tokenizer.
    - timesteps válidos ficam no intervalo [0, max_timesteps - 1].
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        emb_dim,
        n_layers,
        n_heads,
        max_timesteps,
        dropout=0.1,
        tie_weights=False,
    ):
        super().__init__()

        if vocab_size < 2:
            raise ValueError("vocab_size precisa ser >= 2.")
        if block_size < 1:
            raise ValueError("block_size precisa ser >= 1.")
        if emb_dim < 1:
            raise ValueError("emb_dim precisa ser >= 1.")
        if n_layers < 1:
            raise ValueError("n_layers precisa ser >= 1.")
        if n_heads < 1:
            raise ValueError("n_heads precisa ser >= 1.")
        if emb_dim % n_heads != 0:
            raise ValueError("emb_dim precisa ser divisível por n_heads.")
        if max_timesteps < 2:
            raise ValueError("max_timesteps precisa ser >= 2.")

        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.emb_dim = int(emb_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_timesteps = int(max_timesteps)
        self.dropout = float(dropout)
        self.tie_weights = bool(tie_weights)

        self.MASK = self.vocab_size - 1

        # Embeddings compartilhados
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.block_size, self.emb_dim)
        self.time_emb = nn.Embedding(self.max_timesteps, self.emb_dim)

        # Backbone compartilhado
        layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.n_heads,
            dim_feedforward=self.emb_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.n_layers)

        # Cabeça de saída
        self.lm_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

        if self.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)

        if not self.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _validate_token_tensor(self, x_ids, name="x_ids"):
        if not torch.is_tensor(x_ids):
            raise TypeError(f"{name} precisa ser um torch.Tensor.")
        if x_ids.ndim != 2:
            raise ValueError(f"{name} precisa ter shape (B, T), mas veio {tuple(x_ids.shape)}.")
        if x_ids.dtype != torch.long:
            raise TypeError(f"{name} precisa ter dtype torch.long, mas veio {x_ids.dtype}.")

        batch_size, seq_len = x_ids.shape

        if batch_size < 1:
            raise ValueError(f"{name} precisa ter batch_size >= 1.")
        if seq_len < 1:
            raise ValueError(f"{name} precisa ter seq_len >= 1.")
        if seq_len > self.block_size:
            raise ValueError(
                f"{name} tem seq_len={seq_len}, mas block_size={self.block_size}."
            )

        min_id = int(x_ids.min().item())
        max_id = int(x_ids.max().item())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"{name} contém ids fora do range [0, {self.vocab_size - 1}]. "
                f"min={min_id}, max={max_id}."
            )

        return batch_size, seq_len

    def _validate_targets(self, targets, expected_shape):
        if targets is None:
            return

        if not torch.is_tensor(targets):
            raise TypeError("targets precisa ser um torch.Tensor.")
        if targets.ndim != 2:
            raise ValueError(f"targets precisa ter shape (B, T), mas veio {tuple(targets.shape)}.")
        if targets.dtype != torch.long:
            raise TypeError(f"targets precisa ter dtype torch.long, mas veio {targets.dtype}.")
        if tuple(targets.shape) != tuple(expected_shape):
            raise ValueError(
                f"targets precisa ter shape {tuple(expected_shape)}, mas veio {tuple(targets.shape)}."
            )

        min_id = int(targets.min().item())
        max_id = int(targets.max().item())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"targets contém ids fora do range [0, {self.vocab_size - 1}]. "
                f"min={min_id}, max={max_id}."
            )

    def _validate_timesteps(self, t, batch_size, device):
        if not torch.is_tensor(t):
            raise TypeError("t precisa ser um torch.Tensor.")
        if t.ndim != 1:
            raise ValueError(f"t precisa ter shape (B,), mas veio {tuple(t.shape)}.")
        if t.size(0) != batch_size:
            raise ValueError(
                f"t precisa ter batch_size={batch_size}, mas veio {t.size(0)}."
            )

        t = t.to(device=device, dtype=torch.long)

        min_t = int(t.min().item())
        max_t = int(t.max().item())
        if min_t < 0 or max_t >= self.max_timesteps:
            raise ValueError(
                f"t contém valores fora do range [0, {self.max_timesteps - 1}]. "
                f"min={min_t}, max={max_t}."
            )

        return t

    def _positions(self, seq_len, device):
        return torch.arange(seq_len, device=device, dtype=torch.long)

    def _masked_cross_entropy(self, logits, targets, mask=None):
        if mask is not None and bool(mask.any()):
            return F.cross_entropy(logits[mask], targets[mask])

        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

    def _project(self, hidden_states):
        return self.lm_head(hidden_states)

    def forward_diffusion(self, x_ids, t, targets=None):
        """
        x_ids  : (B, T) sequência parcialmente mascarada
        t      : (B,)   timestep de ruído
        targets: (B, T) sequência original
        """
        batch_size, seq_len = self._validate_token_tensor(x_ids, "x_ids")
        self._validate_targets(targets, (batch_size, seq_len))
        t = self._validate_timesteps(t, batch_size, x_ids.device)

        tok = self.tok_emb(x_ids)
        pos = self.pos_emb(self._positions(seq_len, x_ids.device))[None, :, :]
        time = self.time_emb(t)[:, None, :]

        hidden = tok + pos + time
        hidden = self.transformer(hidden)
        logits = self._project(hidden)

        loss = None
        if targets is not None:
            mask = (x_ids == self.MASK)
            loss = self._masked_cross_entropy(logits, targets, mask=mask)

        return logits, loss

    def forward_ar(self, x_ids, targets=None):
        """
        x_ids  : (B, T) sequência de entrada
        targets: (B, T) sequência alvo

        A loss AR usa deslocamento causal:
        posição i prevê o token i+1.
        """
        batch_size, seq_len = self._validate_token_tensor(x_ids, "x_ids")
        self._validate_targets(targets, (batch_size, seq_len))

        tok = self.tok_emb(x_ids)
        pos = self.pos_emb(self._positions(seq_len, x_ids.device))[None, :, :]
        causal_mask = self._causal_mask(seq_len, x_ids.device)

        hidden = tok + pos
        hidden = self.transformer(hidden, mask=causal_mask)
        logits = self._project(hidden)

        loss = None
        if targets is not None:
            if seq_len < 2:
                loss = logits.new_zeros(())
            else:
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, self.vocab_size),
                    targets[:, 1:].reshape(-1),
                )

        return logits, loss
