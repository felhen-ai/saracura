"""Benchmark v1: mesmas fontes do v0, com mais dados de treino, split de validação e descrições de classe.

Diferenças em relação ao v0 (build_bench.py):
- fontes de split único são divididas 80/20 (antes 50/50); teste limitado a 1.000 por tarefa; treino sem limite;
- 10% do treino (até 300 por tarefa) vira validação (`dev`), para escolher configuração sem tocar no teste;
- `camara_tema` ganha treino extra de `ronunes/LegiSubject-Br-Summaries` (MIT), só proposições de tema único;
- a descrição de cada opção vem de data/bench/descriptions.json (geradas pelo Qwen 3.8 27B a partir do treino).

Uso: python build_bench_v1.py
"""

import ast
import json
import random
from collections import Counter

import build_bench as v0
from datasets import load_dataset

OUT = v0.ROOT / "data" / "bench_v1"
SEED = v0.SEED
MAX_TEST, MAX_DEV, MAX_CAMARA_EXTRA = 1000, 300, 12000


def shuffled(rows, salt):
    rows = rows[:]
    random.Random(SEED + salt).shuffle(rows)
    return rows


def split80(rows):
    rows = shuffled(rows, 11)
    k = int(len(rows) * 0.2)
    return rows[k:], rows[:k]


def build():
    desc = json.loads((v0.ROOT / "data" / "bench" / "descriptions.json").read_text())
    tasks = {}

    olid = load_dataset("dougtrajano/olid-br")
    tasks["olid_ofensivo"] = tuple(
        v0.noul_rows(
            [(r["text"], r["is_offensive"] == "OFF") for r in olid[s]],
            "Este comentário contém linguagem ofensiva?",
        )
        for s in ("train", "test")
    )
    alvo = {
        "IND": "um indivíduo específico",
        "GRP": "um grupo de pessoas",
        "OTH": "outro alvo (organização, evento, tema)",
    }
    tasks["olid_alvo"] = tuple(
        v0.choice_rows(
            [(r["text"], r["targeted_type"]) for r in olid[s] if r["targeted_type"] in alvo],
            "A quem a ofensa deste comentário é direcionada?",
            alvo,
        )
        for s in ("train", "test")
    )

    fact = load_dataset("MTEB-BR/factckbr")
    tasks["factck_veracidade"] = tuple(
        v0.choice_rows(
            [(r["text"], r["label"]) for r in fact[s]],
            "Segundo a checagem de fatos, como esta alegação é classificada?",
        )
        for s in ("train", "test")
    )

    faq = load_dataset("ruanchaves/faquad-nli", revision="refs/convert/parquet")

    def pair(r):
        return (f"Pergunta: {r['question']}\nTrecho: {r['answer']}", int(r["label"]) == 1)

    tasks["faquad_resposta"] = (
        v0.noul_rows(
            [pair(r) for s in ("train", "validation") for r in faq[s]],
            "O trecho responde adequadamente à pergunta?",
        ),
        v0.noul_rows([pair(r) for r in faq["test"]], "O trecho responde adequadamente à pergunta?"),
    )

    sci = load_dataset("MTEB-BR/scielo-clustering")["test"]
    tasks["scielo_area"] = split80(
        v0.choice_rows(
            [(r["sentences"], r["label"]) for r in sci],
            "A qual grande área do conhecimento este resumo científico pertence?",
            v0.SCIELO_PT,
        )
    )
    tcu = load_dataset("MTEB-BR/juristcu-clustering")["test"]
    tasks["juristcu_area"] = split80(
        v0.choice_rows(
            [(r["sentences"], r["label"]) for r in tcu],
            "A qual área da jurisprudência do TCU este excerto pertence?",
        )
    )

    cam = load_dataset("MTEB-BR/camara-proposicoes-clustering")["test"]
    cam_train, cam_test = split80(
        v0.choice_rows(
            [(r["sentences"], r["label"]) for r in cam],
            "Qual é o tema desta proposição legislativa?",
        )
    )
    criteria = cam_train[0]["criteria"]
    extra = []
    for r in load_dataset("ronunes/LegiSubject-Br-Summaries", "fold0", split="train"):
        subjects = ast.literal_eval(r["subject"]) if isinstance(r["subject"], str) else r["subject"]
        text = v0.norm(r["summary"])
        if (
            len(subjects) == 1
            and v0.slug(subjects[0]) in criteria
            and text
            and text != text.upper()
        ):
            extra.append(
                {
                    "text": text,
                    "type": "choice",
                    "instructions": cam_train[0]["instructions"],
                    "criteria": criteria,
                    "gold": v0.slug(subjects[0]),
                }
            )
    tasks["camara_tema"] = (cam_train + shuffled(extra, 23)[:MAX_CAMARA_EXTRA], cam_test)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (train, test) in tasks.items():
        test = shuffled(test, 1)[:MAX_TEST]
        held = {r["text"] for r in test}
        seen, clean = set(), []
        for r in shuffled(train, 2):
            if r["text"] not in held and r["text"] not in seen:
                seen.add(r["text"])
                clean.append(r)
        n_dev = min(MAX_DEV, len(clean) // 10)
        dev, train = clean[:n_dev], clean[n_dev:]
        for split, rows in (("train", train), ("dev", dev), ("test", test)):
            with (OUT / f"{name}.{split}.jsonl").open("w") as f:
                for r in rows:
                    if r["criteria"] and name in desc:
                        r = {
                            **r,
                            "criteria": {k: desc[name].get(k, v) for k, v in r["criteria"].items()},
                        }
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gold = Counter(str(r["gold"]) for r in test)
        src = dict(v0.SOURCES)
        src["camara_tema"] = (
            src["camara_tema"][0] + " + ronunes/LegiSubject-Br-Summaries",
            "CC BY 4.0 / MIT; ato oficial",
        )
        manifest[name] = {
            "source": src[name][0],
            "license": src[name][1],
            "type": test[0]["type"],
            "options": len(test[0]["criteria"]) if test[0]["criteria"] else 2,
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "majority_test": round(gold.most_common(1)[0][1] / len(test), 4),
        }
        print(
            name,
            {k: manifest[name][k] for k in ("options", "train", "dev", "test", "majority_test")},
        )
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    build()
