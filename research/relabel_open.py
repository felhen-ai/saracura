"""Refaz os rótulos de `segmento` com um modelo aberto Apache-2.0 servido localmente (vLLM).

O modelo nunca vê os rótulos anteriores (Opus); eles são usados apenas depois, para medir concordância.
A chave do endpoint local é lida do env do serviço dentro do processo e nunca é impressa.

Uso (na VM, como root): python relabel_open.py --n 200            # piloto
                         python relabel_open.py                    # tudo
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from segmento import MAX_CHARS, ROOT, SEGMENTOS, load_rows

BASE = "http://127.0.0.1:8182/v1"
ENV_FILES = [
    Path("/etc/pulso-gpu-arbiter/felhen-coding-runtime.env"),
    Path("/etc/pulso-gpu-arbiter/felhen-coding-gateway.env"),
]
SYSTEM = (
    "Você classifica anúncios de autopeças de marketplaces brasileiros em exatamente um segmento. "
    "Responda apenas com o nome do segmento, exatamente como listado.\n\nSegmentos:\n"
    + "\n".join(f"- {k}: {v}" for k, v in SEGMENTOS.items())
)


def _get(path, key):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def find_key():
    candidates = []
    for f in ENV_FILES:
        if f.exists():
            for line in f.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    name, value = line.split("=", 1)
                    if "KEY" in name.upper() or "TOKEN" in name.upper():
                        candidates.append(value.strip().strip('"').strip("'"))
    for value in candidates:
        try:
            return value, _get("/models", value)["data"][0]["id"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError):
            continue
    raise SystemExit("nenhuma credencial do endpoint local funcionou")


def classify(text, key, model, mode):
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 12,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:MAX_CHARS]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if mode == "structured_outputs":
        body["structured_outputs"] = {"choice": list(SEGMENTOS)}
    else:
        body["guided_choice"] = list(SEGMENTOS)
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return (out["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    key, model = find_key()
    rows = load_rows()
    if args.n:
        rows = rows[: args.n]

    mode = "structured_outputs"
    try:
        first = classify(rows[0]["text"], key, model, mode)
    except urllib.error.HTTPError:
        mode = "guided_choice"
        first = classify(rows[0]["text"], key, model, mode)
    print(f"modelo={model} modo={mode} primeira_resposta_valida={first in SEGMENTOS}", flush=True)

    def work(row):
        for _ in range(3):
            try:
                return classify(row["text"], key, model, mode)
            except Exception:
                time.sleep(2)
        return None

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        labels = list(pool.map(work, rows))
    dt = time.time() - t0

    valid = [(r, label) for r, label in zip(rows, labels, strict=False) if label in SEGMENTOS]
    agree = sum(r["segmento"] == label for r, label in valid) / max(1, len(valid))
    summary = {
        "model": model,
        "mode": mode,
        "rows": len(rows),
        "valid": len(valid),
        "agreement_with_previous_labels": round(agree, 4),
        "seconds": round(dt, 1),
        "ms_per_item_wall": round(1000 * dt / len(rows), 1),
        "workers": args.workers,
        "new_label_distribution": dict(Counter(label for _, label in valid).most_common()),
    }
    suffix = f"_n{args.n}" if args.n else ""
    out = ROOT / "data" / f"relabel_open{suffix}.jsonl"
    with out.open("w") as f:
        for i, label in enumerate(labels):
            f.write(json.dumps({"i": i, "segmento": label if label in SEGMENTOS else None}) + "\n")
    (ROOT / "results" / f"relabel_open{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
