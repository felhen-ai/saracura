"""FAQ Bacen com o LLM local (Qwen 3.8 27B via vLLM), mesmo plano do faq_bacen.py. Rodar como root."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_llm import chat
from faq_bacen import build_plan
from relabel_open import find_key

ROOT = Path(__file__).resolve().parent
key, model = find_key()
plan = build_plan()
SYSTEM = (
    "Você recebe uma pergunta de FAQ do Banco Central e quatro trechos (a, b, c, d). Responda só com a letra "
    "do trecho que começa a responder corretamente à pergunta."
)


def one(row):
    opts = "\n".join(f"{k}) {c}" for k, c in zip("abcd", row["choices"], strict=True))
    out = chat(
        key,
        model,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Pergunta: {row['query']}\n\n{opts}"},
        ],
        list("abcd"),
        max_tokens=4,
    )
    return out == "abcd"[row["gold"]], out is None


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, plan))
correct = sum(r[0] for r in res)
out = {
    "model": "Qwen/Qwen3.8-27B (NVFP4)",
    "accuracy": round(correct / len(plan), 4),
    "correct": correct,
    "rows": len(plan),
    "failed": sum(r[1] for r in res),
}
(ROOT / "results" / "ptbr_board" / "llm27b_bacen.json").write_text(json.dumps(out, indent=1))
print("LLM_BACEN", out)
