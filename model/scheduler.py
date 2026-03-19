# model/scheduler.py

import torch
import math


class NoiseScheduler:
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def get_prob(self, t):
        base = 0.5 * (1 - torch.cos(math.pi * t / self.max_steps))
        return base * 0.5  # 🔥 LIMITADO

    def add_noise(self, x, t, mask_token):
        prob = self.get_prob(t).unsqueeze(1)
        rand = torch.rand_like(x.float())

        mask = rand < prob
        x_noisy = x.clone()
        x_noisy[mask] = mask_token

        return x_noisy