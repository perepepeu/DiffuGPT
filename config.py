# config.py

import torch

class Config:
    vocab_size    = 1024
    block_size    = 128

    emb_dim       = 256
    n_layers      = 6
    n_heads       = 8

    max_timesteps = 8

    batch_size    = 16
    lr            = 2e-4
    epochs        = 30

    # Hybrid: peso da loss AR vs Diffusion
    # 0.0 = só diffusion | 1.0 = só AR | 0.5 = igual
    ar_alpha      = 0.5

    # AR Refiner (inferência)
    ar_temperature = 1.0
    ar_threshold   = 0.65
    ar_top_k       = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"
