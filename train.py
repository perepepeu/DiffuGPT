import os
import json
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import Config
from model.hybrid_model import HybridModel
from model.scheduler import NoiseScheduler
from data.dataset import TextDataset


def save_live_logs(train_losses, val_losses, diff_losses, ar_losses):
    os.makedirs("loss", exist_ok=True)
    data = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "diff_loss": diff_losses,
        "ar_loss": ar_losses,
    }
    with open("loss/live.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_final_logs(train_losses, val_losses, diff_losses, ar_losses):
    os.makedirs("loss", exist_ok=True)
    data = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "diff_loss": diff_losses,
        "ar_loss": ar_losses,
    }
    with open("loss/final.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cfg_value(cfg, name, default):
    return getattr(cfg, name, default)


def make_dirs():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("loss", exist_ok=True)


def build_model(cfg, device):
    return HybridModel(
        cfg.vocab_size,
        cfg.block_size,
        cfg.emb_dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.max_timesteps,
    ).to(device)


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_compiled_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_existing_weights(model, device):
    if not os.path.exists("model.pt"):
        return

    ckpt = safe_torch_load("model.pt", map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    state_dict = strip_compiled_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Aviso: chaves ausentes ao carregar model.pt: {missing}")
    if unexpected:
        print(f"Aviso: chaves inesperadas ao carregar model.pt: {unexpected}")

    print("Pesos de model.pt carregados com sucesso.")


def save_best_weights(model):
    torch.save(model.state_dict(), "model.pt")


def save_epoch_checkpoint(model, optimizer, epoch, best_val, train_losses, val_losses, diff_losses, ar_losses):
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "diff_loss": diff_losses,
        "ar_loss": ar_losses,
    }
    torch.save(ckpt, f"checkpoints/epoch_{epoch}.pt")


def validate_batch(batch, cfg):
    if batch.ndim != 2:
        raise ValueError(f"Batch precisa ter shape (B, T), mas veio {tuple(batch.shape)}.")

    if batch.size(1) != cfg.block_size:
        raise ValueError(
            f"Batch com tamanho de sequência {batch.size(1)}, mas cfg.block_size={cfg.block_size}."
        )

    min_id = int(batch.min().item())
    max_id = int(batch.max().item())

    if min_id < 0 or max_id >= cfg.vocab_size:
        raise ValueError(
            f"IDs fora do vocabulário detectados. min={min_id}, max={max_id}, vocab_size={cfg.vocab_size}."
        )


def sample_timesteps(batch_size, max_timesteps, device, generator=None):
    if max_timesteps < 2:
        raise ValueError("cfg.max_timesteps precisa ser >= 2.")

    t = torch.randint(1, max_timesteps, (batch_size,), generator=generator, device="cpu")
    return t.to(device)


def add_noise(batch, t, mask_token, scheduler, generator=None):
    prob = scheduler.get_prob(t).unsqueeze(1)

    if generator is None:
        rand = torch.rand(batch.shape, device=batch.device, dtype=torch.float32)
    else:
        rand = torch.rand(batch.shape, generator=generator, device="cpu", dtype=torch.float32).to(batch.device)

    mask = rand < prob

    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        row_ids = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)

        if generator is None:
            col_ids = torch.randint(0, batch.size(1), (row_ids.numel(),), device=batch.device)
        else:
            col_ids = torch.randint(
                0, batch.size(1), (row_ids.numel(),), generator=generator, device="cpu"
            ).to(batch.device)

        mask[row_ids, col_ids] = True

    x_noisy = batch.clone()
    x_noisy[mask] = mask_token
    return x_noisy


def make_loaders(dataset, cfg, seed, pin_memory):
    if len(dataset) < 2:
        raise ValueError("O dataset precisa ter pelo menos 2 amostras para split train/val.")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0:
        train_size = 1
        val_size = len(dataset) - 1

    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    split_gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)

    num_workers = int(get_cfg_value(cfg, "num_workers", 0))

    train_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    val_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    if len(train_loader) == 0:
        raise ValueError("train_loader ficou vazio.")
    if len(val_loader) == 0:
        raise ValueError("val_loader ficou vazio.")

    return train_loader, val_loader


def main():
    cfg = Config()
    seed = int(get_cfg_value(cfg, "seed", 42))
    grad_clip = float(get_cfg_value(cfg, "grad_clip", 1.0))
    live_log_every = int(get_cfg_value(cfg, "live_log_every", 50))
    weight_decay = float(get_cfg_value(cfg, "weight_decay", 0.01))
    use_compile = bool(get_cfg_value(cfg, "use_compile", True))

    device = torch.device(cfg.device)
    pin_memory = device.type == "cuda"

    seed_everything(seed)
    make_dirs()

    dataset = TextDataset("data/train_ids.npy")
    train_loader, val_loader = make_loaders(dataset, cfg, seed, pin_memory)

    base_model = build_model(cfg, device)
    load_existing_weights(base_model, device)

    model = base_model
    if use_compile:
        try:
            model = torch.compile(base_model)
            print("torch.compile ativado.")
        except Exception as e:
            model = base_model
            print(f"torch.compile indisponível: {e}. Seguindo sem compile.")

    scheduler = NoiseScheduler(cfg.max_timesteps)
    optimizer = optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    best_val = float("inf")
    train_losses, val_losses = [], []
    diff_losses, ar_losses = [], []

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_diff = 0.0
        total_ar = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for step_idx, batch in enumerate(pbar, start=1):
            batch = torch.as_tensor(batch, dtype=torch.long, device=device)
            validate_batch(batch, cfg)

            t = sample_timesteps(batch.size(0), cfg.max_timesteps, device=device)
            x_noisy = add_noise(batch, t, base_model.MASK, scheduler)

            _, loss_diff = model.forward_diffusion(x_noisy, t, targets=batch)
            _, loss_ar = model.forward_ar(batch, targets=batch)
            loss = (1.0 - cfg.ar_alpha) * loss_diff + cfg.ar_alpha * loss_ar

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss inválida no treino. loss={loss.item()}, diff={loss_diff.item()}, ar={loss_ar.item()}."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_diff += float(loss_diff.item())
            total_ar += float(loss_ar.item())
            steps += 1

            avg_loss = total_loss / steps
            avg_diff = total_diff / steps
            avg_ar = total_ar / steps

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                diff=f"{avg_diff:.4f}",
                ar=f"{avg_ar:.4f}",
            )

            if step_idx == 1 or step_idx % max(1, live_log_every) == 0 or step_idx == len(train_loader):
                save_live_logs(
                    train_losses + [avg_loss],
                    val_losses,
                    diff_losses + [avg_diff],
                    ar_losses + [avg_ar],
                )

        if steps == 0:
            raise RuntimeError("Nenhum passo de treino foi executado.")

        train_loss = total_loss / steps
        epoch_diff = total_diff / steps
        epoch_ar = total_ar / steps

        model.eval()
        val_total = 0.0
        val_diff_total = 0.0
        val_ar_total = 0.0
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = torch.as_tensor(batch, dtype=torch.long, device=device)
                validate_batch(batch, cfg)

                val_gen = torch.Generator().manual_seed(seed + batch_idx)
                t = sample_timesteps(batch.size(0), cfg.max_timesteps, device=device, generator=val_gen)
                x_noisy = add_noise(batch, t, base_model.MASK, scheduler, generator=val_gen)

                _, ld = model.forward_diffusion(x_noisy, t, targets=batch)
                _, la = model.forward_ar(batch, targets=batch)
                lv = (1.0 - cfg.ar_alpha) * ld + cfg.ar_alpha * la

                if not torch.isfinite(lv):
                    raise RuntimeError(
                        f"Loss inválida na validação. val={lv.item()}, diff={ld.item()}, ar={la.item()}."
                    )

                val_total += float(lv.item())
                val_diff_total += float(ld.item())
                val_ar_total += float(la.item())
                count += 1

        if count == 0:
            raise RuntimeError("Nenhum passo de validação foi executado.")

        val_loss = val_total / count

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        diff_losses.append(epoch_diff)
        ar_losses.append(epoch_ar)

        save_live_logs(train_losses, val_losses, diff_losses, ar_losses)

        print(
            f"\nEpoch {epoch + 1} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Diff {epoch_diff:.4f} | AR {epoch_ar:.4f} | "
            f"ValDiff {val_diff_total / count:.4f} | ValAR {val_ar_total / count:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_best_weights(base_model)
            print(f"  Melhor modelo salvo (val={val_loss:.4f})")

        save_epoch_checkpoint(
            base_model,
            optimizer,
            epoch + 1,
            best_val,
            train_losses,
            val_losses,
            diff_losses,
            ar_losses,
        )

    save_final_logs(train_losses, val_losses, diff_losses, ar_losses)

    if os.path.exists("loss/live.json"):
        os.remove("loss/live.json")

    print("\nTreinamento finalizado.")


if __name__ == "__main__":
    main()
