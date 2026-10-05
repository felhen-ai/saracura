import sys

import laya
from huggingface_hub import HfApi

tok = sys.stdin.readline().strip()
api = HfApi(token=tok)
ds = api.dataset_info("felhen-ai/ptbr-typed-decisions-bench")
print(
    "dataset privado:",
    ds.private,
    "| arquivos:",
    len(ds.siblings),
    "| configs no card:",
    len((ds.card_data or {}).get("configs", [])),
)
mi = api.model_info("felhen-ai/saracura-ptbr-v0")
print("modelo privado:", mi.private, "| arquivos:", [s.rfilename for s in mi.siblings][:8])
agent = laya.load("felhen-ai/saracura-ptbr-v0", token=tok, device="cuda")
res = agent.predict(
    "Requer informações ao Ministro da Saúde sobre a distribuição de vacinas nos municípios do interior.",
    {
        "tema": {
            "type": "choice",
            "instructions": "Qual é o tema desta proposição legislativa?",
            "criteria": {"saude": "Saúde", "educacao": "Educação", "economia": "Economia"},
        },
        "pede_informacao": {
            "type": "noul",
            "instructions": "O texto é um pedido de informação a uma autoridade?",
        },
    },
)
print(
    "teste de carga do Hub:",
    {
        k: (v.get("choice") if "choice" in v else round(v.get("noul"), 3))
        for k, v in res["answers"].items()
    },
    "| max_len:",
    agent.cfg.get("max_len"),
    agent.cfg.get("head_max_len"),
)
