"""Trilha de pesquisa Saracura: decisão `segmento` (choice, 12 opções) em anúncios reais PT-BR.

Mede, no mesmo conjunto de teste, baselines clássicos e checkpoints Laya em zero-shot.
Gabarito: rótulos do Claude Opus do experimento de fine-tuning do PulsoOnline (não é
gabarito humano). Saída: apenas números agregados; nenhuma linha de anúncio é gravada.

Uso: CUDA_VISIBLE_DEVICES="" python eval_segmento.py [--n-test 1000] [--models a,b]
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/home/amello/finetuning/balanced_training_25k.jsonl")
OUT = Path(__file__).resolve().parent / "results"
SEED = 20261002

SEGMENTOS = {
    "Motor": "peças internas e externas do motor: cabeçote, bloco, pistão, biela, virabrequim, correias, juntas, coxim do motor, turbina",
    "Suspensao": "amortecedor, mola, bandeja, pivô, bucha, barra estabilizadora, cubo de roda, rolamento de roda",
    "Freio": "pastilha, disco, tambor, pinça, cilindro mestre, servo freio, módulo ABS, cabo de freio de mão",
    "Eletrica": "alternador, motor de partida, módulos e centrais eletrônicas, sensores, chicotes, faróis, lanternas, chaves, painel de instrumentos, som",
    "Transmissao": "câmbio, embreagem, semieixo, homocinética, diferencial, cardan, trambulador, alavanca de câmbio",
    "Carroceria": "portas, capô, para-lama, para-choque, tampa traseira, vidros, retrovisores, grades, acabamentos externos e internos, bancos",
    "Arrefecimento": "radiador, ventoinha, bomba d'água, reservatório de expansão, válvula termostática, mangueiras, ar-condicionado e compressor",
    "Direcao": "caixa de direção, bomba de direção hidráulica, coluna de direção, volante, terminal e barra de direção",
    "Escapamento": "coletor de escape, catalisador, silencioso, tubo de escapamento, sonda lambda",
    "Combustivel": "bomba de combustível, bico injetor, tanque, flauta, corpo de borboleta (TBI), filtro de combustível, carburador",
    "Acessorios": "acessórios e itens avulsos: tapetes, calotas, rodas, ferramentas, macaco, manuais, capas, engates",
    "Outros": "itens que não se encaixam em nenhum dos outros segmentos de autopeças",
}


def load_rows():
    rows = []
    for line in DATA.open():
        d = json.loads(line)
        user = next(m["content"] for m in d["messages"] if m["role"] == "user")
        gold = json.loads(next(m["content"] for m in d["messages"] if m["role"] == "assistant"))
        seg = gold.get("segmento")
        if seg in SEGMENTOS:
            rows.append({"text": user, "segmento": seg})
    return rows


def split(rows, n_test):
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in rows:
        by[r["segmento"]].append(r)
    test, train = [], []
    for _seg, items in sorted(by.items()):
        rng.shuffle(items)
        k = round(n_test * len(items) / len(rows))
        test += items[:k]
        train += items[k:]
    rng.shuffle(test)
    rng.shuffle(train)
    return train, test


def metrics(gold, pred):
    acc = sum(g == p for g, p in zip(gold, pred, strict=False)) / len(gold)
    per = {}
    for seg in SEGMENTOS:
        idx = [i for i, g in enumerate(gold) if g == seg]
        per[seg] = round(sum(pred[i] == seg for i in idx) / len(idx), 4) if idx else None
    macro = sum(v for v in per.values() if v is not None) / sum(v is not None for v in per.values())
    return {
        "accuracy": round(acc, 4),
        "macro_recall": round(macro, 4),
        "per_segmento": per,
        "pred_distribution": dict(Counter(pred).most_common()),
    }


def run_classic(train, test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    gold = [r["segmento"] for r in test]
    majority = Counter(r["segmento"] for r in train).most_common(1)[0][0]
    out = {"majority": metrics(gold, [majority] * len(test))}
    for n in (200, 1000, len(train)):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
            LogisticRegression(max_iter=2000, C=10.0),
        )
        sub = train[:n]
        t0 = time.time()
        clf.fit([r["text"] for r in sub], [r["segmento"] for r in sub])
        pred = list(clf.predict([r["text"] for r in test]))
        m = metrics(gold, pred)
        m["train_examples"] = len(sub)
        m["seconds"] = round(time.time() - t0, 1)
        out[f"tfidf_logreg_n{len(sub)}"] = m
    return out


def run_laya(name, model_id, subfolder, test, max_chars):
    import laya

    agent = laya.load(model_id, subfolder=subfolder, device="cpu")
    question = {
        "segmento": {
            "type": "choice",
            "instructions": "A qual segmento de autopeças este anúncio pertence?",
            "criteria": SEGMENTOS,
        }
    }
    states = [r["text"][:max_chars] for r in test]
    t0 = time.time()
    results = agent.predict_batch(states, question, batch_size=8)
    dt = time.time() - t0
    pred = [res["answers"]["segmento"]["choice"] for res in results]
    truncated = sum(
        bool(res["usage"]["truncated"] or res["usage"]["truncated_questions"]) for res in results
    )
    m = metrics([r["segmento"] for r in test], pred)
    m.update(
        {
            "model": model_id,
            "subfolder": subfolder,
            "device": "cpu",
            "truncated_items": truncated,
            "ms_per_item": round(1000 * dt / len(test), 1),
            "max_chars": max_chars,
        }
    )
    return m


LAYA_MODELS = {
    "laya_multilingual_zeroshot": ("convaiinnovations/laya", "multilingual"),
    "telepatia_laya_pt_es_typed": ("telepatia-ai/laya-pt-es-typed", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--models", default="classic," + ",".join(LAYA_MODELS))
    ap.add_argument("--max-chars", type=int, default=600)
    args = ap.parse_args()

    rows = load_rows()
    train, test = split(rows, args.n_test)
    report = {
        "task": "segmento/choice/12",
        "seed": SEED,
        "rows": len(rows),
        "train": len(train),
        "test": len(test),
        "chance": round(1 / len(SEGMENTOS), 4),
        "gold": "rótulos Claude Opus (experimento PulsoOnline, mar/2026); não é gabarito humano",
        "results": {},
    }
    OUT.mkdir(exist_ok=True)
    for name in args.models.split(","):
        if name == "classic":
            report["results"].update(run_classic(train, test))
        else:
            model_id, sub = LAYA_MODELS[name]
            try:
                report["results"][name] = run_laya(name, model_id, sub, test, args.max_chars)
            except Exception as exc:  # registra a falha em vez de derrubar as outras medições
                report["results"][name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        (OUT / "segmento_baselines.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1)
        )
        r = report["results"]
        for k in r:
            print(
                k,
                {
                    kk: r[k].get(kk)
                    for kk in ("accuracy", "macro_recall", "ms_per_item", "error")
                    if kk in r[k]
                },
                flush=True,
            )
        print("---", flush=True)


if __name__ == "__main__":
    main()
