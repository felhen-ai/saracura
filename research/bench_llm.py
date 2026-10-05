"""Usa o LLM aberto local (Qwen 3.8 27B, Apache-2.0, via vLLM) sobre o benchmark.

  python bench_llm.py ceiling        # acerto do LLM sem ajuste em cada tarefa (teto de referência)
  python bench_llm.py describe       # gera uma descrição por classe das tarefas choice -> data/bench/descriptions.json

Rodar como root na VM (a chave do endpoint local é lida do env do serviço dentro do processo).
"""

import argparse
import json
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import bench
from relabel_open import BASE, find_key

SEED = 20261002


def chat(key, model, messages, choices=None, max_tokens=16):
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if choices:
        body["structured_outputs"] = {"choice": choices}
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return (json.load(r)["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            time.sleep(2 + attempt)
    return None


def cmd_ceiling(_args):
    key, model = find_key()
    descriptions = {}
    path = bench.BENCH / "descriptions.json"
    if path.exists():
        descriptions = json.loads(path.read_text())
    report = {"model": "Qwen/Qwen3.8-27B (NVFP4)", "tasks": {}}
    for task in bench.tasks():
        rows = bench.load(task, "test")

        def work(row, task=task):
            if row["type"] == "noul":
                system = "Responda apenas 'sim' ou 'nao'."
                out = chat(
                    key,
                    model,
                    [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": f"{row['text'][:6000]}\n\n{row['instructions']}",
                        },
                    ],
                    ["sim", "nao"],
                )
                return None if out is None else out == "sim"
            crit = row["criteria"]
            desc = descriptions.get(task, {})
            options = "\n".join(f"- {k}: {desc.get(k) or v}" for k, v in crit.items())
            system = f"{row['instructions']}\nResponda apenas com o identificador de uma das opções.\n\nOpções:\n{options}"
            return chat(
                key,
                model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["text"][:6000]},
                ],
                list(crit),
            )

        t0 = time.time()
        with ThreadPoolExecutor(24) as pool:
            pred = list(pool.map(work, rows))
        m = bench.score([r["gold"] for r in rows], pred)
        m["invalid"] = sum(p is None for p in pred)
        m["seconds"] = round(time.time() - t0, 1)
        report["tasks"][task] = m
        print(task, m, flush=True)
    vals = [m["balanced_accuracy"] for m in report["tasks"].values()]
    report["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
    report["with_descriptions"] = bool(descriptions)
    name = "bench_llm27b_desc.json" if descriptions else "bench_llm27b.json"
    (bench.ROOT / "results" / name).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("MEDIA", report["mean_balanced_accuracy"], "| com descrições:", bool(descriptions))


def cmd_describe(_args):
    key, model = find_key()
    rng = random.Random(SEED)
    out = {}
    for task in bench.tasks():
        train = bench.load(task, "train")
        if train[0]["type"] != "choice":
            continue
        by = defaultdict(list)
        for r in train:
            by[r["gold"]].append(r["text"])
        crit = train[0]["criteria"]
        jobs = []
        for k, name in crit.items():
            examples = rng.sample(by[k], min(6, len(by[k])))
            shown = "\n".join(f"- {e[:400]}" for e in examples)
            others = ", ".join(v for kk, v in crit.items() if kk != k)
            prompt = (
                f"Tarefa: {train[0]['instructions']}\nClasse: {name}\nOutras classes: {others}\n\n"
                f"Exemplos de textos desta classe:\n{shown}\n\n"
                "Escreva uma única frase, em português do Brasil, com até 25 palavras, que descreva o que caracteriza "
                "os textos desta classe e a distingue das outras. Comece pelo nome da classe seguido de dois-pontos. "
                "Não cite os exemplos literalmente."
            )
            jobs.append((k, prompt))
        with ThreadPoolExecutor(12) as pool:
            texts = list(
                pool.map(
                    lambda j: chat(key, model, [{"role": "user", "content": j[1]}], max_tokens=80),
                    jobs,
                )
            )
        out[task] = {
            k: (t or crit[k]).replace("\n", " ").strip()
            for (k, _), t in zip(jobs, texts, strict=False)
        }
        print(
            task,
            len(out[task]),
            "descrições; ex:",
            next(iter(out[task].values()))[:160],
            flush=True,
        )
    (bench.BENCH / "descriptions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ceiling", "describe"])
    args = ap.parse_args()
    {"ceiling": cmd_ceiling, "describe": cmd_describe}[args.cmd](args)


if __name__ == "__main__":
    main()
