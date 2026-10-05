"""Gera perguntas variadas sobre textos reais e as responde com o LLM aberto local (professor), para
ensinar o modelo pequeno a ler perguntas novas em vez de reconhecer tarefas fixas.

  python gen_questions.py generate --name train   --per-task 800 --ads 1500   # perguntas + respostas -> data/gen/train.jsonl
  python gen_questions.py generate --name heldout --per-task 60  --ads 120 --seed 7 --split dev
  python gen_questions.py dedupe                                              # remove do treino instruções do held-out
  python gen_questions.py prep --model-dir models/laya/multilingual --out items/gen_train.pt

Cada pergunta é respondida 5 vezes com temperatura 0,7; a distribuição das respostas é o alvo (como no
dataset original do Laya). Rodar como root na VM com o serviço do modelo de 27B ativo.
"""

import argparse
import json
import random
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bench
from relabel_open import BASE, find_key
from segmento import load_rows as load_ads
from segmento import split as split_ads

ROOT = bench.ROOT
GEN = ROOT / "data" / "gen"
N_SAMPLES, TEMP = 5, 0.7

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "perguntas": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "tipo": {"type": "string", "enum": ["escolha", "sim_nao", "escala"]},
                    "instrucao": {"type": "string"},
                    "opcoes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "niveis": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tipo", "instrucao"],
            },
        }
    },
    "required": ["perguntas"],
}

GEN_PROMPT = """Você cria perguntas de decisão sobre um texto, para treinar um sistema que responde perguntas fechadas.

Dado o texto abaixo, escreva de 3 a 5 perguntas variadas, em português do Brasil, que uma pessoa ou um sistema poderia querer responder sobre ele. Misture os três tipos:
- "escolha": a resposta é uma entre 2 a 6 opções; dê "opcoes" como objeto {{identificador_curto: descrição da opção}}.
- "sim_nao": a resposta é sim ou não.
- "escala": a resposta é um nível ordenado; dê "niveis" como lista de 3 a 5 rótulos do menor ao maior.

Regras: as perguntas devem ser respondíveis só com o texto; variem o assunto (intenção, tom, urgência, público, categoria, risco, qualidade, ação recomendada, presença de algo); não repita o tema entre perguntas; não faça perguntas cuja resposta seja óbvia pelo tipo do texto. Responda só com JSON no formato {{"perguntas": [...]}}.

Texto:
\"\"\"{text}\"\"\""""


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


def chat(key, model, messages, *, choices=None, schema=None, max_tokens=16, n=1, temperature=0.0):
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "n": n,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if choices:
        body["structured_outputs"] = {"choice": choices}
    if schema:
        body["structured_outputs"] = {"json": schema}
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return [c["message"].get("content") or "" for c in json.load(r)["choices"]]
        except Exception:
            time.sleep(2 + attempt)
    return None


def pool(per_task, ads, seed, split):
    rng = random.Random(seed)
    texts = []
    for task in bench.tasks():
        rows = bench.load(task, split)
        rng.shuffle(rows)
        texts += [(task, r["text"][:2500]) for r in rows[:per_task]]
    train, test = split_ads(load_ads())
    rows = train if split == "train" else test
    rng.shuffle(rows)
    texts += [("anuncios", r["text"][:2500]) for r in rows[:ads]]
    rng.shuffle(texts)
    return texts


def to_question(p):
    if p["tipo"] == "escolha":
        opts = {
            norm(k).replace(" ", "_")[:30]: v for k, v in (p.get("opcoes") or {}).items() if k and v
        }
        if not 2 <= len(opts) <= 8:
            return None
        return {"type": "choice", "instructions": p["instrucao"], "criteria": opts}
    if p["tipo"] == "sim_nao":
        return {"type": "noul", "instructions": p["instrucao"]}
    levels = [level for level in (p.get("niveis") or []) if level]
    if not 3 <= len(levels) <= 5:
        return None
    return {"type": "score", "instructions": p["instrucao"], "criteria": levels}


def answer(key, model, text, q):
    system = "Responda à pergunta sobre o texto apenas com o identificador de uma das opções."
    if q["type"] == "choice":
        choices = list(q["criteria"])
        opts = "\n".join(f"- {k}: {v}" for k, v in q["criteria"].items())
    elif q["type"] == "noul":
        choices = ["sim", "nao"]
        opts = "- sim\n- nao"
    else:
        choices = [str(i) for i in range(len(q["criteria"]))]
        opts = "\n".join(f"- {i}: {lvl}" for i, lvl in enumerate(q["criteria"]))
    user = f'Texto:\n"""{text}"""\n\nPergunta: {q["instructions"]}\nOpções:\n{opts}\n\nResponda só com o identificador, sem explicação.'
    # Sem gramática por requisição: a compilação de uma gramática nova a cada pergunta travava a CPU da VM.
    outs = chat(
        key,
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        n=N_SAMPLES,
        temperature=TEMP,
        max_tokens=12,
    )
    if not outs:
        return None

    def parse(o):
        o = norm(o).strip(" .:-\"'`")
        if o in choices:
            return o
        for c in choices:  # aceita "a) ...", "sim." etc.
            if (
                o.startswith(c + " ")
                or o.startswith(c + ")")
                or o.startswith(c + ":")
                or o.startswith(c + ".")
            ):
                return c
        return None

    counts = Counter(p for p in map(parse, outs) if p)
    if not counts:
        return None
    total = sum(counts.values())
    if q["type"] == "noul":
        probs = {"true": counts["sim"] / total, "false": counts["nao"] / total}
    else:
        probs = {c: counts[c] / total for c in choices}
    return {"probabilities": probs}


def cmd_generate(args):
    key, model = find_key()
    texts = pool(args.per_task, args.ads, args.seed, args.split)
    print(f"textos: {len(texts)}", flush=True)

    def work(item):
        source, text = item
        outs = chat(
            key,
            model,
            [{"role": "user", "content": GEN_PROMPT.format(text=text)}],
            schema=QUESTION_SCHEMA,
            max_tokens=700,
            temperature=0.8,
        )
        if not outs:
            return None
        try:
            perguntas = json.loads(outs[0])["perguntas"]
        except (ValueError, KeyError, TypeError):
            return None
        questions, gold = {}, {}
        for i, p in enumerate(perguntas):
            q = to_question(p) if isinstance(p, dict) and p.get("instrucao") else None
            if not q:
                continue
            g = answer(key, model, text, q)
            if g:
                questions[f"q{i}"] = q
                gold[f"q{i}"] = g
        return (
            {"source": source, "state": text, "questions": questions, "gold": gold}
            if questions
            else None
        )

    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / f"{args.name}.jsonl"
    t0 = time.time()
    n_rows = n_q = 0
    with ThreadPoolExecutor(args.workers) as pool_, out.open("w") as f:
        for i, row in enumerate(pool_.map(work, texts)):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_q += len(row["questions"])
            if (i + 1) % 200 == 0:
                print(
                    f"{i + 1}/{len(texts)} textos, {n_q} perguntas, {round(time.time() - t0)} s",
                    flush=True,
                )
    types = Counter(
        q["type"] for line in out.open() for q in json.loads(line)["questions"].values()
    )
    print(
        f"gravado {out}: {n_rows} textos, {n_q} perguntas, tipos={dict(types)}, {round(time.time() - t0)} s"
    )


def cmd_dedupe(_args):
    held = {
        norm(q["instructions"])
        for line in (GEN / "heldout.jsonl").open()
        for q in json.loads(line)["questions"].values()
    }
    kept = dropped = 0
    rows = [json.loads(line) for line in (GEN / "train.jsonl").open()]
    with (GEN / "train.jsonl").open("w") as f:
        for row in rows:
            qs = {k: q for k, q in row["questions"].items() if norm(q["instructions"]) not in held}
            dropped += len(row["questions"]) - len(qs)
            if qs:
                row["questions"] = qs
                row["gold"] = {k: row["gold"][k] for k in qs}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += len(qs)
    print(
        f"treino: {kept} perguntas mantidas, {dropped} removidas por coincidir com o held-out ({len(held)} instruções)"
    )


def cmd_prep(args):
    import torch
    from laya_finetune_mps import build_training_item
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    cfg = {**cfg, "max_len": bench.MAX_LEN, "head_max_len": bench.HEAD_MAX_LEN}
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    items, skipped = [], 0
    for line in (GEN / args.name).open():
        row = json.loads(line)
        for qid, q in row["questions"].items():
            item = build_training_item(tokenizer, cfg, row["state"], q, row["gold"][qid])
            if item is None:
                skipped += 1
            else:
                items.append(item)
    random.Random(bench.SEED).shuffle(items)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, args.out)
    print(f"itens={len(items)} pulados={skipped} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--name", required=True)
    g.add_argument("--per-task", type=int, default=800)
    g.add_argument("--ads", type=int, default=1500)
    g.add_argument("--seed", type=int, default=bench.SEED)
    g.add_argument("--split", default="train")
    g.add_argument("--workers", type=int, default=24)
    sub.add_parser("dedupe")
    p = sub.add_parser("prep")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="train.jsonl")
    args = ap.parse_args()
    {"generate": cmd_generate, "dedupe": cmd_dedupe, "prep": cmd_prep}[args.cmd](args)


if __name__ == "__main__":
    main()
