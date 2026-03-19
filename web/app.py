# web/app.py

import torch
import os
import json
from flask import Flask, render_template, request, jsonify

from config import Config
from model.hybrid_model import HybridModel
from tokenizer import BPEFastTokenizer
from sample import generate, generate_ar, generate_hybrid

app = Flask(__name__)
cfg = Config()

tok = BPEFastTokenizer()
tok.load("data/tok-1024.model")


def load_model(path):
    model = HybridModel(
        cfg.vocab_size, cfg.block_size, cfg.emb_dim,
        cfg.n_layers,   cfg.n_heads,    cfg.max_timesteps,
    ).to(cfg.device)

    state_dict = torch.load(path, map_location=cfg.device)

    # remove prefixo _orig_mod. gerado pelo torch.compile()
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


current_model = None


@app.route("/checkpoints")
def checkpoints():
    files = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
    files.sort()
    return jsonify(files)


@app.route("/load_model", methods=["POST"])
def load_model_api():
    global current_model
    data          = request.json
    name          = data["name"]
    path          = f"checkpoints/{name}" if name != "best" else "model.pt"
    current_model = load_model(path)
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate_api():
    global current_model

    if current_model is None:
        return jsonify({"error": "Nenhum modelo carregado."}), 400

    data       = request.json
    prompt     = data["prompt"]
    mode       = data.get("mode", "diffusion")   # "diffusion" | "ar" | "hybrid"
    min_tokens = int(data.get("min_tokens", 32))
    min_tokens = max(8, min(min_tokens, cfg.block_size))

    # Modo 1: Diffusion pura
    if mode == "diffusion":
        show = data.get("diffusion", False)
        out, tokens, tps = generate(
            current_model, tok, prompt,
            steps=24,
            min_new_tokens=min_tokens,
            show_steps=show,
            inline=False,
        )
        return jsonify({"text": out, "tokens": tokens, "tps": round(tps, 2), "mode": "diffusion"})

    # Modo 2: AR puro
    elif mode == "ar":
        temperature = float(data.get("temperature", cfg.ar_temperature))
        top_k       = int(data.get("top_k", cfg.ar_top_k))
        out, tokens, tps = generate_ar(
            current_model, tok, prompt,
            max_new_tokens=min_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return jsonify({"text": out, "tokens": tokens, "tps": round(tps, 2), "mode": "ar"})

    # Modo 3: Hibrido
    elif mode == "hybrid":
        temperature = float(data.get("temperature", cfg.ar_temperature))
        threshold   = float(data.get("threshold",   cfg.ar_threshold))
        show        = data.get("diffusion", False)

        draft, refined, dt, dtps, at, atps = generate_hybrid(
            current_model, tok, prompt,
            diff_steps=24,
            min_new_tokens=min_tokens,
            ar_temperature=temperature,
            ar_threshold=threshold,
            show_steps=show,
        )
        return jsonify({
            "draft":       draft,
            "text":        refined,
            "diff_tokens": dt,
            "diff_tps":    round(dtps, 2),
            "ar_tokens":   at,
            "ar_tps":      round(atps, 2),
            "mode":        "hybrid",
        })

    return jsonify({"error": f"Modo invalido: {mode}"}), 400


@app.route("/logs")
def logs():
    for path in ["loss/final.json", "loss/live.json", "logs.json"]:
        if os.path.exists(path):
            with open(path) as f:
                return jsonify(json.load(f))
    return jsonify({"train_loss": [], "val_loss": [], "diff_loss": [], "ar_loss": []})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
