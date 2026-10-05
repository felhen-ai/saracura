"""Executa o script upstream de fine-tuning do Laya em CUDA, sem alterar o arquivo upstream.

O upstream só escolhe entre MPS e CPU; aqui a escolha de device é substituída por CUDA.
Uso: mesmos argumentos de laya_finetune_mps.py.
"""

import os

import laya_finetune_mps as upstream
import torch

upstream.choose_device = lambda _requested: torch.device("cuda")

# LR_SCALE multiplica as taxas de aprendizado fixas do upstream (encoder 2.5e-5, cabeça 1e-4).
_scale = float(os.environ.get("LR_SCALE", "1"))
if _scale != 1:
    _adamw = torch.optim.AdamW

    def _scaled(groups, **kwargs):
        return _adamw([{**g, "lr": g["lr"] * _scale} for g in groups], **kwargs)

    torch.optim.AdamW = _scaled

if __name__ == "__main__":
    upstream.main()
