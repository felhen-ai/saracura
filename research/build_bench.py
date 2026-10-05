"""Monta o benchmark v0 de decisões tipadas em PT-BR a partir de fontes abertas de licença permissiva.

Cada tarefa vira dois arquivos em data/bench/: <tarefa>.train.jsonl e <tarefa>.test.jsonl, com linhas
{"text", "type" (choice|noul), "instructions", "criteria" (dict id->descrição, só choice), "gold"}.
Fontes e licenças (verificadas na origem em 02/10/2026) ficam em SOURCES e vão para o manifesto.

Uso: python build_bench.py
"""

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "bench"
SEED = 20261002
MAX_TRAIN, MAX_TEST = 3000, 500

SOURCES = {
    "olid_ofensivo": ("dougtrajano/olid-br", "CC BY 4.0"),
    "olid_alvo": ("dougtrajano/olid-br", "CC BY 4.0"),
    "factck_veracidade": ("MTEB-BR/factckbr (FACTCK.BR)", "MIT / Apache-2.0"),
    "faquad_resposta": ("ruanchaves/faquad-nli (FaQuAD)", "CC BY 4.0"),
    "scielo_area": ("MTEB-BR/scielo-clustering", "CC BY 4.0"),
    "juristcu_area": ("MTEB-BR/juristcu-clustering", "CC BY 4.0"),
    "camara_tema": (
        "MTEB-BR/camara-proposicoes-clustering (Dados Abertos da Câmara)",
        "CC BY 4.0 (compilação); ato oficial",
    ),
}

SCIELO_PT = {
    "Agricultural Sciences": "Ciências Agrárias",
    "Biological/Life Sciences": "Ciências Biológicas",
    "Engineering & Technology": "Engenharias e Tecnologia",
    "Health Sciences": "Ciências da Saúde",
    "Humanities & Arts": "Humanidades e Artes",
    "Mathematics & Computer Science": "Matemática e Ciência da Computação",
    "Physical Sciences & Chemistry": "Ciências Físicas e Química",
    "Social Sciences": "Ciências Sociais",
}


def slug(label):
    s = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def choice_rows(pairs, instructions, rename=None):
    """pairs: (texto, rótulo). Monta criteria com todos os rótulos observados."""
    labels = sorted({label for _, label in pairs})
    criteria = {slug(label): (rename or {}).get(label, label) for label in labels}
    return [
        {
            "text": norm(t),
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
            "gold": slug(label),
        }
        for t, label in pairs
        if norm(t)
    ]


def noul_rows(pairs, instructions):
    return [
        {
            "text": norm(t),
            "type": "noul",
            "instructions": instructions,
            "criteria": None,
            "gold": bool(g),
        }
        for t, g in pairs
        if norm(t)
    ]


def split_rows(rows, frac_test=0.5):
    rng = random.Random(SEED)
    rows = rows[:]
    rng.shuffle(rows)
    k = int(len(rows) * frac_test)
    return rows[k:], rows[:k]


def cap(rows, n):
    rng = random.Random(SEED + 1)
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:n]


def build():
    tasks = {}

    olid = load_dataset("dougtrajano/olid-br")
    ins = "Este comentário contém linguagem ofensiva?"
    tasks["olid_ofensivo"] = tuple(
        noul_rows([(r["text"], r["is_offensive"] == "OFF") for r in olid[s]], ins)
        for s in ("train", "test")
    )
    alvo = {
        "IND": "um indivíduo específico",
        "GRP": "um grupo de pessoas",
        "OTH": "outro alvo (organização, evento, tema)",
    }
    ins = "A quem a ofensa deste comentário é direcionada?"
    tasks["olid_alvo"] = tuple(
        choice_rows(
            [(r["text"], r["targeted_type"]) for r in olid[s] if r["targeted_type"] in alvo],
            ins,
            alvo,
        )
        for s in ("train", "test")
    )

    fact = load_dataset("MTEB-BR/factckbr")
    ins = "Segundo a checagem de fatos, como esta alegação é classificada?"
    tasks["factck_veracidade"] = tuple(
        choice_rows([(r["text"], r["label"]) for r in fact[s]], ins) for s in ("train", "test")
    )

    faq = load_dataset("ruanchaves/faquad-nli", revision="refs/convert/parquet")
    ins = "O trecho responde adequadamente à pergunta?"
    tasks["faquad_resposta"] = tuple(
        noul_rows(
            [
                (f"Pergunta: {r['question']}\nTrecho: {r['answer']}", int(r["label"]) == 1)
                for r in faq[s]
            ],
            ins,
        )
        for s in ("train", "test")
    )

    sci = load_dataset("MTEB-BR/scielo-clustering")["test"]
    tasks["scielo_area"] = split_rows(
        choice_rows(
            [(r["sentences"], r["label"]) for r in sci],
            "A qual grande área do conhecimento este resumo científico pertence?",
            SCIELO_PT,
        )
    )

    tcu = load_dataset("MTEB-BR/juristcu-clustering")["test"]
    tasks["juristcu_area"] = split_rows(
        choice_rows(
            [(r["sentences"], r["label"]) for r in tcu],
            "A qual área da jurisprudência do TCU este excerto pertence?",
        )
    )

    cam = load_dataset("MTEB-BR/camara-proposicoes-clustering")["test"]
    tasks["camara_tema"] = split_rows(
        choice_rows(
            [(r["sentences"], r["label"]) for r in cam],
            "Qual é o tema desta proposição legislativa?",
        )
    )

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (train, test) in tasks.items():
        test_texts = {r["text"] for r in test}
        train = [r for r in train if r["text"] not in test_texts]
        train, test = cap(train, MAX_TRAIN), cap(test, MAX_TEST)
        for split, rows in (("train", train), ("test", test)):
            with (OUT / f"{name}.{split}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gold = Counter(str(r["gold"]) for r in test)
        manifest[name] = {
            "source": SOURCES[name][0],
            "license": SOURCES[name][1],
            "type": test[0]["type"],
            "options": len(test[0]["criteria"]) if test[0]["criteria"] else 2,
            "train": len(train),
            "test": len(test),
            "majority_test": round(gold.most_common(1)[0][1] / len(test), 4),
        }
        print(name, manifest[name])
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    build()
