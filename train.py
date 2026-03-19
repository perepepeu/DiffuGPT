# train.py

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import os
import time
import json

from config import Config
from model.hybrid_model import HybridModel
from model.scheduler import NoiseScheduler
from data.dataset import TextDataset


def save_live_logs(train_losses, val_losses, diff_losses, ar_losses):
    os.makedirs("loss", exist_ok=True)
    data = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "diff_loss":  diff_losses,
        "ar_loss":    ar_losses,
    }
    with open("loss/live.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    cfg = Config()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("loss", exist_ok=True)

    dataset    = TextDataset("data/train_ids.npy")
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size)

    model = HybridModel(
        cfg.vocab_size, cfg.block_size, cfg.emb_dim,
        cfg.n_layers,   cfg.n_heads,    cfg.max_timesteps,
    ).to(cfg.device)

    # carrega pesos existentes se houver (compativel com DiffusionModel)
    if os.path.exists("model.pt"):
        try:
            model.load_state_dict(torch.load("model.pt", map_location=cfg.device))
            print("Pesos de model.pt carregados com sucesso.")
        except Exception as e:
            print(f"Nao foi possivel carregar model.pt: {e}. Iniciando do zero.")

    try:
        model = torch.compile(model)
    except Exception:
        pass

    scheduler = NoiseScheduler(cfg.max_timesteps)
    opt       = optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    train_losses, val_losses = [], []
    diff_losses,  ar_losses  = [], []

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = total_diff = total_ar = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            batch = batch.to(cfg.device)

            # loss de difusao
            t       = torch.randint(1, cfg.max_timesteps, (batch.size(0),), device=cfg.device)
            x_noisy = scheduler.add_noise(batch, t, model.MASK)
            _, loss_diff = model.forward_diffusion(x_noisy, t, targets=batch)

            # loss AR (GPT-style, texto limpo)
            _, loss_ar = model.forward_ar(batch, targets=batch)

            # combinacao
            loss = (1 - cfg.ar_alpha) * loss_diff + cfg.ar_alpha * loss_ar

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_diff += loss_diff.item()
            total_ar   += loss_ar.item()
            steps += 1

            pbar.set_postfix(
                loss=f"{total_loss/steps:.4f}",
                diff=f"{total_diff/steps:.4f}",
                ar=f"{total_ar/steps:.4f}",
            )

            save_live_logs(
                train_losses + [total_loss / steps],
                val_losses,
                diff_losses  + [total_diff / steps],
                ar_losses    + [total_ar   / steps],
            )

        train_loss = total_loss / steps
        epoch_diff = total_diff / steps
        epoch_ar   = total_ar   / steps

        # validacao
        model.eval()
        val_total = val_diff_total = val_ar_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(cfg.device)

                t       = torch.randint(1, cfg.max_timesteps, (batch.size(0),), device=cfg.device)
                x_noisy = scheduler.add_noise(batch, t, model.MASK)
                _, ld   = model.forward_diffusion(x_noisy, t, targets=batch)
                _, la   = model.forward_ar(batch, targets=batch)
                lv      = (1 - cfg.ar_alpha) * ld + cfg.ar_alpha * la

                val_total      += lv.item()
                val_diff_total += ld.item()
                val_ar_total   += la.item()
                count += 1

        val_loss = val_total / count

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        diff_losses.append(epoch_diff)
        ar_losses.append(epoch_ar)

        save_live_logs(train_losses, val_losses, diff_losses, ar_losses)

        print(
            f"\nEpoch {epoch+1} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Diff {epoch_diff:.4f} | AR {epoch_ar:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "model.pt")
            print(f"  Melhor modelo salvo (val={val_loss:.4f})")

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")

    with open("loss/final.json", "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss":   val_losses,
            "diff_loss":  diff_losses,
            "ar_loss":    ar_losses,
        }, f, indent=2)

    if os.path.exists("loss/live.json"):
        os.remove("loss/live.json")

    print("\nTreinamento finalizado.")


if __name__ == "__main__":
    main()
