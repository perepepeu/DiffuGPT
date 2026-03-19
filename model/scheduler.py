# model/scheduler.py

import math
import torch


class NoiseScheduler:
    """
    Scheduler de ruído para masked discrete diffusion.

    Estratégia:
    - usa uma curva cosseno para aumentar a probabilidade de máscara ao longo do tempo;
    - pode garantir ao menos 1 token mascarado por sequência;
    - mantém API compatível com o restante do projeto.
    """

    def __init__(
        self,
        max_steps,
        min_mask_prob=0.0,
        max_mask_prob=0.80,
        ensure_one_mask=True,
    ):
        if max_steps < 2:
            raise ValueError("max_steps precisa ser >= 2.")
        if not (0.0 <= min_mask_prob <= 1.0):
            raise ValueError("min_mask_prob precisa estar entre 0.0 e 1.0.")
        if not (0.0 <= max_mask_prob <= 1.0):
            raise ValueError("max_mask_prob precisa estar entre 0.0 e 1.0.")
        if min_mask_prob > max_mask_prob:
            raise ValueError("min_mask_prob não pode ser maior que max_mask_prob.")

        self.max_steps = int(max_steps)
        self.min_mask_prob = float(min_mask_prob)
        self.max_mask_prob = float(max_mask_prob)
        self.ensure_one_mask = bool(ensure_one_mask)

    def _to_float_tensor(self, t, device=None):
        if torch.is_tensor(t):
            return t.to(device=device, dtype=torch.float32)
        return torch.tensor(t, device=device, dtype=torch.float32)

    def get_prob(self, t):
        """
        Retorna probabilidade de máscara para cada timestep.

        Intervalo esperado de t:
        [0, max_steps - 1]
        """
        t = self._to_float_tensor(t)

        min_t = float(t.min().item())
        max_t = float(t.max().item())

        if min_t < 0 or max_t > (self.max_steps - 1):
            raise ValueError(
                f"t fora do range [0, {self.max_steps - 1}]. "
                f"min={min_t}, max={max_t}."
            )

        if self.max_steps == 1:
            progress = torch.zeros_like(t)
        else:
            progress = t / float(self.max_steps - 1)

        cosine = 0.5 * (1.0 - torch.cos(math.pi * progress))
        prob = self.min_mask_prob + (self.max_mask_prob - self.min_mask_prob) * cosine
        return prob.clamp(0.0, 1.0)

    def add_noise(self, x, t, mask_token, return_mask=False):
        """
        x         : (B, T) ids originais
        t         : (B,)   timestep por item do batch
        mask_token: id reservado para máscara

        Retorna:
        - x_noisy
        - opcionalmente mask, se return_mask=True
        """
        if not torch.is_tensor(x):
            raise TypeError("x precisa ser um torch.Tensor.")
        if x.ndim != 2:
            raise ValueError(f"x precisa ter shape (B, T), mas veio {tuple(x.shape)}.")
        if x.dtype != torch.long:
            raise TypeError(f"x precisa ter dtype torch.long, mas veio {x.dtype}.")
        if x.size(0) < 1 or x.size(1) < 1:
            raise ValueError("x precisa ter shape (B, T) com B >= 1 e T >= 1.")

        if not isinstance(mask_token, int):
            mask_token = int(mask_token)
        if mask_token < 0:
            raise ValueError("mask_token precisa ser >= 0.")

        if not torch.is_tensor(t):
            raise TypeError("t precisa ser um torch.Tensor.")
        if t.ndim != 1:
            raise ValueError(f"t precisa ter shape (B,), mas veio {tuple(t.shape)}.")
        if t.size(0) != x.size(0):
            raise ValueError(
                f"t precisa ter batch_size={x.size(0)}, mas veio {t.size(0)}."
            )

        t = t.to(device=x.device, dtype=torch.long)
        prob = self.get_prob(t).to(x.device).unsqueeze(1)

        rand = torch.rand(x.shape, device=x.device, dtype=torch.float32)
        mask = rand < prob

        if self.ensure_one_mask:
            empty_rows = ~mask.any(dim=1)
            if bool(empty_rows.any()):
                rows = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
                cols = torch.randint(0, x.size(1), (rows.numel(),), device=x.device)
                mask[rows, cols] = True

        x_noisy = x.clone()
        x_noisy[mask] = mask_token

        if return_mask:
            return x_noisy, mask

        return x_noisy
