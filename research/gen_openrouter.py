"""Gera perguntas variadas e respostas com probabilidades usando um professor grande de pesos abertos via OpenRouter.

Duas chamadas por texto: (1) escrever de 3 a 5 perguntas em JSON; (2) responder todas de uma vez, distribuindo
probabilidades entre as opções. Só textos de fontes abertas e anúncios; nada interno. A chave vem de OPENROUTER_API_KEY
no ambiente (carregada de ~/.env pelo chamador, sem eco). Orçamento com teto em dólares e aviso a cada US$ 10.

  python gen_openrouter.py --name pilot_or --per-task 12 --ads 16              # piloto (~100 textos)
  python gen_openrouter.py --name train_or --per-task 800 --ads 1500 --budget 25
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "data" / "bench_v1"
GEN = ROOT / "data" / "gen_or"
MODEL = "qwen/qwen3.5-397b-a17b"  # Apache-2.0 (Qwen/Qwen3.5-397B-A17B), conferido em 04/10/2026
PRICE_IN, PRICE_OUT = 0.55 / 1e6, 3.50 / 1e6
URL = "https://openrouter.ai/api/v1/chat/completions"
SEED = 20261002

GEN_PROMPT = """Você cria perguntas de decisão sobre um texto, para treinar um sistema que responde perguntas fechadas.

Dado o texto abaixo, escreva de 3 a 5 perguntas variadas, em português do Brasil, que uma pessoa ou um sistema poderia querer responder sobre ele. Misture os três tipos:
- "escolha": a resposta é uma entre 2 a 6 opções; dê "opcoes" como objeto {{identificador_curto: descrição da opção}}.
- "sim_nao": a resposta é sim ou não.
- "escala": a resposta é um nível ordenado; dê "niveis" como lista de 3 a 5 rótulos do menor ao maior.

Regras: as perguntas devem ser respondíveis só com o texto; variem o assunto (intenção, tom, urgência, público, categoria, risco, qualidade, ação recomendada, presença de algo); não repita o tema entre perguntas; não faça perguntas cuja resposta seja óbvia pelo tipo do texto. Responda só com JSON no formato {{"perguntas": [{{"tipo": ..., "instrucao": ..., "opcoes": {{...}} ou "niveis": [...]}}]}}.

Texto:
\"\"\"{text}\"\"\""""

ANSWER_PROMPT = """Responda às perguntas sobre o texto. Para cada pergunta, distribua 100% de probabilidade entre as opções listadas, refletindo sua confiança (concentre a massa na opção correta quando o texto for claro; espalhe quando for ambíguo). Use exatamente os identificadores dados.

Texto:
\"\"\"{text}\"\"\"

Perguntas:
{questions}

Responda só com JSON no formato {{"respostas": {{"q0": {{"id_opcao": probabilidade, ...}}, "q1": {{...}}}}}}."""


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


class Budget:
    def __init__(self, limit):
        self.limit, self.spent, self.calls, self.lock, self.next_warn = (
            limit,
            0.0,
            0,
            threading.Lock(),
            10.0,
        )

    def add(self, usage):
        cost = (
            usage.get("prompt_tokens", 0) * PRICE_IN + usage.get("completion_tokens", 0) * PRICE_OUT
        )
        with self.lock:
            self.spent += cost
            self.calls += 1
            if self.spent >= self.next_warn:
                print(
                    f"### gasto acumulado US$ {self.spent:.2f} em {self.calls} chamadas", flush=True
                )
                self.next_warn += 10.0
            return self.spent < self.limit


def chat(key, budget, prompt, max_tokens, temperature):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "provider": {"data_collection": "deny"},
        "usage": {"include": True},
        "reasoning": {"enabled": False},
    }  # sem isso o Qwen 3.5 gasta centenas de tokens de raciocínio por chamada
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/felhen-ai/saracura",
            "X-Title": "Saracura PT-BR question generation",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)
            if not budget.add(out.get("usage") or {}):
                raise SystemExit(f"teto de US$ {budget.limit} atingido")
            return out["choices"][0]["message"].get("content") or ""
        except SystemExit:
            raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def parse_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except ValueError:
            return None


def to_question(p):
    if not isinstance(p, dict) or not p.get("instrucao"):
        return None
    if p.get("tipo") == "escolha":
        opts = {
            norm(k).replace(" ", "_")[:30]: str(v)
            for k, v in (p.get("opcoes") or {}).items()
            if k and v
        }
        return (
            {"type": "choice", "instructions": p["instrucao"], "criteria": opts}
            if 2 <= len(opts) <= 8
            else None
        )
    if p.get("tipo") == "sim_nao":
        return {"type": "noul", "instructions": p["instrucao"]}
    if p.get("tipo") == "escala":
        levels = [str(level) for level in (p.get("niveis") or []) if level]
        return (
            {"type": "score", "instructions": p["instrucao"], "criteria": levels}
            if 3 <= len(levels) <= 5
            else None
        )
    return None


def render_questions(questions):
    lines = []
    for qid, q in questions.items():
        if q["type"] == "choice":
            opts = "; ".join(f"{k}: {v}" for k, v in q["criteria"].items())
        elif q["type"] == "noul":
            opts = "sim; nao"
        else:
            opts = "; ".join(f"{i}: {lvl}" for i, lvl in enumerate(q["criteria"]))
        lines.append(f"{qid} [{q['type']}] {q['instructions']}\n   opções: {opts}")
    return "\n".join(lines)


def to_gold(q, dist):
    if not isinstance(dist, dict):
        return None
    if q["type"] == "choice":
        keys = list(q["criteria"])
    elif q["type"] == "noul":
        keys = ["sim", "nao"]
    else:
        keys = [str(i) for i in range(len(q["criteria"]))]
    probs = {}
    for k, v in dist.items():
        k = norm(str(k)).replace(" ", "_")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if k in keys and v >= 0:
            probs[k] = probs.get(k, 0) + v
    total = sum(probs.values())
    if total <= 0 or max(probs.values()) / total < 0.34:
        return None
    probs = {k: probs.get(k, 0) / total for k in keys}
    if q["type"] == "noul":
        probs = {"true": probs["sim"], "false": probs["nao"]}
    return {"probabilities": probs}


def pool(per_task, ads, seed):
    rng = random.Random(seed)
    texts = []
    for f in sorted(BENCH.glob("*.train.jsonl")):
        rows = [json.loads(line) for line in f.open()]
        rng.shuffle(rows)
        texts += [(f.name.split(".")[0], r["text"][:2500]) for r in rows[:per_task]]
    if ads:
        sys.path.insert(0, str(ROOT))
        from segmento import load_rows, split

        train, _ = split(load_rows())
        rng.shuffle(train)
        texts += [("anuncios", r["text"][:2500]) for r in train[:ads]]
    rng.shuffle(texts)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-task", type=int, default=800)
    ap.add_argument("--ads", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY ausente no ambiente")
    budget = Budget(args.budget)
    texts = pool(args.per_task, args.ads, args.seed)
    print(f"modelo={MODEL} textos={len(texts)} teto=US$ {args.budget}", flush=True)

    def work(item):
        source, text = item
        gen = parse_json(chat(key, budget, GEN_PROMPT.format(text=text), 900, 0.8))
        if not gen:
            return None
        questions = {}
        for i, p in enumerate((gen.get("perguntas") or [])[:5]):
            q = to_question(p)
            if q:
                questions[f"q{i}"] = q
        if not questions:
            return None
        ans = parse_json(
            chat(
                key,
                budget,
                ANSWER_PROMPT.format(text=text, questions=render_questions(questions)),
                500,
                0.0,
            )
        )
        resp = (ans or {}).get("respostas") or {}
        gold = {qid: g for qid, q in questions.items() if (g := to_gold(q, resp.get(qid)))}
        questions = {qid: questions[qid] for qid in gold}
        return (
            {
                "source": source,
                "state": text,
                "questions": questions,
                "gold": gold,
                "teacher": MODEL,
            }
            if gold
            else None
        )

    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / f"{args.name}.jsonl"
    if out.exists():  # retomada: pula textos já gravados e continua no mesmo arquivo
        done = {json.loads(line)["state"] for line in out.open()}
        texts = [t for t in texts if t[1] not in done]
        print(f"retomando: {len(done)} textos já gravados, {len(texts)} restantes", flush=True)
    t0, n_rows, n_q = time.time(), 0, 0
    try:
        with ThreadPoolExecutor(args.workers) as ex, out.open("a") as f:
            for i, row in enumerate(ex.map(work, texts)):
                if row:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n_rows += 1
                    n_q += len(row["questions"])
                if (i + 1) % 100 == 0:
                    print(
                        f"{i + 1}/{len(texts)} textos, {n_q} perguntas, US$ {budget.spent:.2f}, {round(time.time() - t0)} s",
                        flush=True,
                    )
    finally:
        types = Counter(
            q["type"] for line in out.open() for q in json.loads(line)["questions"].values()
        )
        print(
            f"gravado {out}: {n_rows} textos, {n_q} perguntas, tipos={dict(types)}, US$ {budget.spent:.2f} em {budget.calls} chamadas, {round(time.time() - t0)} s"
        )


if __name__ == "__main__":
    main()
