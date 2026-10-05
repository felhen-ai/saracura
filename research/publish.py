"""Publica o benchmark e o modelo no Hugging Face, sob a organização felhen-ai, como repositórios PRIVADOS.

O token vem pela entrada padrão (ex.: `op read op://... | ssh vm 'python publish.py dataset'`) e nunca é gravado.

  python publish.py dataset   # data/bench_v1 -> felhen-ai/ptbr-typed-decisions-bench
  python publish.py model     # runs/v1_e4   -> felhen-ai/saracura-ptbr-v0
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
ORG = "felhen-ai"
DATASET_REPO = f"{ORG}/ptbr-typed-decisions-bench"
MODEL_REPO = f"{ORG}/saracura-ptbr-v0"
BENCH = ROOT / "data" / "bench_v1"
RUN = ROOT / "runs" / "v4b"
MAX_LEN, HEAD_MAX_LEN = 2048, 1024


def publish_dataset(api):
    manifest = json.loads((BENCH / "manifest.json").read_text())
    configs = []
    for task in sorted(manifest):
        configs.append(
            {
                "config_name": task,
                "data_files": [
                    {"split": split, "path": f"data/{task}/{split}.jsonl"}
                    for split in ("train", "dev", "test")
                ],
            }
        )
    api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for task in manifest:
            (tmp / "data" / task).mkdir(parents=True)
            for split in ("train", "dev", "test"):
                shutil.copy(BENCH / f"{task}.{split}.jsonl", tmp / "data" / task / f"{split}.jsonl")
        shutil.copy(BENCH / "manifest.json", tmp / "manifest.json")
        card = (ROOT / "publish" / "bench_card.md").read_text()
        head, body = card.split("---\n", 2)[1:]
        head = (
            head.rstrip()
            + "\nconfigs:\n"
            + "".join(
                f"  - config_name: {c['config_name']}\n    data_files:\n"
                + "".join(
                    f"      - split: {d['split']}\n        path: {d['path']}\n"
                    for d in c["data_files"]
                )
                for c in configs
            )
        )
        (tmp / "README.md").write_text(f"---\n{head}\n---\n{body}")
        api.upload_folder(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            folder_path=str(tmp),
            commit_message="Benchmark v1: 7 tarefas, splits train/dev/test, manifesto e card",
        )
    print("dataset publicado (privado):", f"https://huggingface.co/datasets/{DATASET_REPO}")


def publish_model(api):
    api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("model.safetensors", "encoder", "tokenizer"):
            src = RUN / name
            (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp / name)
        cfg = json.loads((RUN / "rl_agent_config.json").read_text())
        cfg.update(
            {"model_name": "saracura-ptbr-v0.1", "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}
        )
        (tmp / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
        shutil.copy(ROOT / "publish" / "model_card.md", tmp / "README.md")
        api.upload_folder(
            repo_id=MODEL_REPO,
            repo_type="model",
            folder_path=str(tmp),
            commit_message="Saracura PT-BR v0.1: perguntas variadas e casos de uso no treino",
        )
    print("modelo publicado (privado):", f"https://huggingface.co/{MODEL_REPO}")


def main():
    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("token ausente na entrada padrão")
    api = HfApi(token=token)
    who = api.whoami()
    assert any(o["name"] == ORG for o in who.get("orgs", [])), "token sem acesso à organização"
    {"dataset": publish_dataset, "model": publish_model}[sys.argv[1]](api)


if __name__ == "__main__":
    main()
