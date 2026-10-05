"""Concordância em `segmento` entre o Laya ajustado e o Qwen 32B no holdout do experimento do Pulso.

O holdout (1.000 anúncios) não tem gabarito do Opus; mede-se concordância entre modelos:
Laya ajustado x Qwen ajustado (QLoRA), x Qwen base, e TF-IDF x Qwen ajustado como referência.
Itens do holdout cujo título aparece no treino são excluídos.

Uso: python agree_qwen.py --model runs/ft_full
"""

import argparse
import json
from pathlib import Path

from segmento import MAX_CHARS, ROOT, SEGMENTOS, load_rows, question, split

FT = Path("/home/amello/finetuning")


def render(item):
    """Mesmo formato do texto de treino: título seguido da descrição."""
    text = f"Titulo: {item['title']}"
    if item.get("description_clean"):
        text += f"\nDescricao: {item['description_clean']}"
    return text


def agreement(a, b):
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x in SEGMENTOS and y in SEGMENTOS]
    return {"n": len(pairs), "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import laya
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train, _ = split(load_rows())
    train_titles = {r["text"].split("\n")[0] for r in train}
    holdout = [json.loads(line) for line in (FT / "finetuning_holdout.jsonl").open()]
    qwen_ft = {
        r["item_id"]: r.get("segmento")
        for r in map(json.loads, (FT / "eval_finetuned.jsonl").open())
    }
    holdout = [h for h in holdout if f"Titulo: {h['title']}" not in train_titles]
    texts = [render(h) for h in holdout]

    agent = laya.load(args.model, device=args.device)
    results = agent.predict_batch(
        [t[:MAX_CHARS] for t in texts], {"segmento": question()}, batch_size=8
    )
    laya_pred = [r["answers"]["segmento"]["choice"] for r in results]

    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(max_iter=2000, C=10.0),
    )
    clf.fit([r["text"] for r in train], [r["segmento"] for r in train])
    tfidf_pred = list(clf.predict(texts))

    q_ft = [qwen_ft.get(h["item_id"]) for h in holdout]
    q_base = [h.get("qwen_segmento") for h in holdout]
    out = {
        "holdout_usado": len(holdout),
        "model": args.model,
        "laya_ft_x_qwen_ft": agreement(laya_pred, q_ft),
        "laya_ft_x_qwen_base": agreement(laya_pred, q_base),
        "tfidf_x_qwen_ft": agreement(tfidf_pred, q_ft),
        "qwen_ft_x_qwen_base": agreement(q_ft, q_base),
        "laya_ft_x_tfidf": agreement(laya_pred, tfidf_pred),
    }
    strata = sorted({h["strata"] for h in holdout})
    out["laya_ft_x_qwen_ft_por_strata"] = {
        s: agreement(
            [p for p, h in zip(laya_pred, holdout, strict=False) if h["strata"] == s],
            [q for q, h in zip(q_ft, holdout, strict=False) if h["strata"] == s],
        )
        for s in strata
    }
    (ROOT / "results" / "agree_qwen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
