"""Converte o split de treino de `segmento` em itens tokenizados para o script de fine-tuning do Laya.

Reusa `build_training_item` do script upstream (laya_finetune_mps.py, Apache-2.0) sem alterá-lo.
A ordem das opções é embaralhada por item para o modelo não aprender posição como classe.

Uso: python prep_items.py --model-dir models/laya/multilingual --out items/train_n4000.pt --n 4000
"""

import argparse
import json
import random
from pathlib import Path

import torch
from laya_finetune_mps import build_training_item
from segmento import MAX_CHARS, SEED, SEGMENTOS, load_rows, question, split
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="quantidade de exemplos de treino (0 = todos)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    cfg = {**cfg, "max_len": cfg.get("max_len", 1024), "head_max_len": cfg.get("head_max_len", 256)}
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    train, _ = split(load_rows())
    if args.n:
        train = train[: args.n]

    rng = random.Random(SEED)
    items, skipped = [], 0
    for row in train:
        order = list(SEGMENTOS)
        rng.shuffle(order)
        item = build_training_item(
            tokenizer,
            cfg,
            row["text"][:MAX_CHARS],
            question(order),
            {"probabilities": {row["segmento"]: 1.0}},
        )
        if item is None:
            skipped += 1
        else:
            items.append(item)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, out)
    lens = sorted(len(i["ids"]) for i in items)
    print(
        f"itens={len(items)} pulados={skipped} tokens p50={lens[len(lens) // 2]} max={lens[-1]} -> {out}"
    )


if __name__ == "__main__":
    main()
