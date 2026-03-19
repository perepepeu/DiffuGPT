
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
    ar_alpha       = 0.5

    # AR Refiner — modo de refinamento
    # "fast"     : batched, threshold alto  (velocidade maxima, menos correcoes)
    # "balanced" : batched, threshold baixo (meio-termo, mais correcoes, padrao)
    ar_refine_mode      = "balanced"
    ar_threshold_fast   = 0.65   # poucos tokens corrigidos
    ar_threshold_balanced = 0.55 # mais agressivo, compensa falta de cascata

    ar_temperature = 1.0
    ar_top_k       = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"