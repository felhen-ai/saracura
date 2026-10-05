"""Tarefa `segmento` (choice, 12 opções) sobre anúncios reais PT-BR do PulsoOnline.

Dados internos: nenhuma linha sai desta pasta; só métricas agregadas são reportadas.
Gabarito: rótulos do Claude Opus (experimento de fine-tuning do Pulso, mar/2026).
"""

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SEGMENTO_DATA", ROOT / "data" / "balanced_training_25k.jsonl"))
SEED = 20261002
MAX_CHARS = 600
INSTRUCTIONS = "A qual segmento de autopeças este anúncio pertence?"

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


def question(order=None):
    keys = order or list(SEGMENTOS)
    return {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {k: SEGMENTOS[k] for k in keys},
    }


def load_rows():
    rows = []
    for line in DATA.open():
        d = json.loads(line)
        user = next(m["content"] for m in d["messages"] if m["role"] == "user")
        gold = json.loads(next(m["content"] for m in d["messages"] if m["role"] == "assistant"))
        if gold.get("segmento") in SEGMENTOS:
            rows.append({"text": user, "segmento": gold["segmento"]})
    return rows


def split(rows, n_test=1000):
    """Split estratificado e determinístico; idêntico ao usado na medição dos baselines."""
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in rows:
        by[r["segmento"]].append(r)
    test, train = [], []
    for _, items in sorted(by.items()):
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
    vals = [v for v in per.values() if v is not None]
    return {
        "accuracy": round(acc, 4),
        "macro_recall": round(sum(vals) / len(vals), 4),
        "per_segmento": per,
        "pred_distribution": dict(Counter(pred).most_common()),
    }
