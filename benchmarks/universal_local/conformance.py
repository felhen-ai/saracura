"""Maintainer-only, descriptor-bound real-tokenizer conformance freezing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import rfc8785

from benchmarks.universal_local.acquisition import VerificationReceipt
from benchmarks.universal_local.inputs import TextChoice, TextState, render_laya, render_nli
from benchmarks.universal_local.loaders import DirectTokenizer, LoadError, _tokenizer
from benchmarks.universal_local.registry import Candidate

ROOT = Path(__file__).parents[2]
ARTIFACT_ROOT = ROOT / ".artifacts" / "universal-local-conformance"
FIXTURE_PATH = ROOT / "benchmarks" / "fixtures" / "universal-local-conformance.v1.json"


def _name(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 64
        and value[0].isalnum()
        and all(character.isalnum() or character in "._-" for character in value)
    )


def _versions() -> dict[str, str]:
    try:
        import safetensors  # type: ignore[import-not-found]
        import tokenizers  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as error:
        raise LoadError("universal-local extra is required") from error
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "tokenizers": str(tokenizers.__version__),
        "safetensors": str(safetensors.__version__),
        "os": f"{platform.system()}-{platform.release()}",
    }


def _scoring_example(candidate: Candidate) -> dict[str, Any]:
    # This is an input-independent public rule vector, not a model output.
    if candidate.id == "laya-multilingual":
        return {
            "rule": "marker_logits_softmax_over_choices",
            "input_marker_logits": [0.0, 1.0],
            "weights": [0.2689414214, 0.7310585786],
        }
    return {
        "rule": "fp32_three_label_softmax_entailment_then_linear_normalization",
        "entailment_index": 0,
        "input_classification_logits": [[0.0, 0.0, 0.0], [1.0986122887, 0.0, 0.0]],
        "weights": [0.3571428571, 0.6428571429],
    }


def _fragment(candidate: Candidate, tokenizer: DirectTokenizer) -> dict[str, Any]:
    public_inputs = (
        (
            "pt-BR",
            TextState(
                "Pedido atualizado",
                "A pessoa pede uma orientação objetiva para acompanhar a solicitação.",
            ),
            "Escolha a orientação mais útil.",
            (
                TextChoice("example-one", "Confirmar o prazo do atendimento."),
                TextChoice("example-two", "Encaminhar para a fila responsável."),
            ),
        ),
        (
            "en",
            TextState("Order update", "The person asks for a clear next step for the request."),
            "Choose the most useful guidance.",
            (
                TextChoice("example-one", "Confirm the stated timeline."),
                TextChoice("example-two", "Route the support request."),
            ),
        ),
    )
    examples: list[dict[str, Any]] = []
    for locale, state, instruction, choices in public_inputs:
        if candidate.id == "laya-multilingual":
            rendered = render_laya(tokenizer, state, instruction, choices)
            if not rendered.status.startswith("supported"):
                raise LoadError("real Laya tokenizer rejected public conformance input")
            examples.append(
                {
                    "locale": locale,
                    "token_ids": list(rendered.input_ids),
                    "marker_positions": list(rendered.marker_positions),
                    "tensor_shapes": {
                        "input_ids": [1, len(rendered.input_ids)],
                        "attention_mask": [1, len(rendered.input_ids)],
                        "marker_pos": [1, len(rendered.marker_positions)],
                        "marker_mask": [1, len(rendered.marker_positions)],
                        "qtype": [1],
                        "marker_logits": [1, len(rendered.marker_positions)],
                        "ranking_weights": [len(rendered.marker_positions)],
                    },
                }
            )
        else:
            rendered_choices = tuple(
                render_nli(tokenizer, state, instruction, choice, locale) for choice in choices
            )
            if not all(item.status.startswith("supported") for item in rendered_choices):
                raise LoadError("real NLI tokenizer rejected public conformance input")
            examples.append(
                {
                    "locale": locale,
                    "token_ids": [list(item.input_ids) for item in rendered_choices],
                    "pair_positions": [list(item.pair_positions) for item in rendered_choices],
                    "tensor_shapes": {
                        "input_ids": [[1, len(item.input_ids)] for item in rendered_choices],
                        "attention_mask": [[1, len(item.input_ids)] for item in rendered_choices],
                        "classification_logits": [2, 3],
                        "ranking_weights": [2],
                    },
                }
            )
    return {
        "candidate_id": candidate.id,
        "acquisition_contract_digest": candidate.acquisition_contract_digest,
        "rendering": "laya_option_marker"
        if candidate.id == "laya-multilingual"
        else "mdeberta_pairwise_nli",
        "examples": examples,
        "output_schema": "uncalibrated_ranking_weights",
        "scoring": _scoring_example(candidate),
        "public_card_example": "PT-BR and English framing",
    }


def freeze_fragment(receipt: VerificationReceipt, candidate: Candidate, output: str) -> Path:
    """Freeze one real-tokenizer vector fragment under the fixed private root."""
    if not _name(output):
        raise ValueError("invalid conformance output")
    ARTIFACT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ARTIFACT_ROOT.is_symlink():
        raise ValueError("invalid conformance artifact root")
    target = ARTIFACT_ROOT / f"{output}.json"
    payload: dict[str, Any] = {
        "schema_version": "universal-local-conformance-fragment.v1",
        "library_versions": _versions(),
        "vector": _fragment(candidate, _tokenizer(receipt, candidate)),
    }
    raw = rfc8785.dumps(payload) + b"\n"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(target)
        raise
    return target


def fragment_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_conformance_receipt(
    receipt: VerificationReceipt, candidate: Candidate
) -> DirectTokenizer:
    """Regenerate the reviewed vector from the live descriptor-bound tokenizer."""
    try:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LoadError("conformance fixture is unavailable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "universal-local-conformance.v1"
        or not isinstance(payload.get("candidates"), list)
    ):
        raise LoadError("conformance fixture is invalid")
    expected = [
        item
        for item in payload["candidates"]
        if isinstance(item, dict) and item.get("candidate_id") == candidate.id
    ]
    if len(expected) != 1:
        raise LoadError("candidate conformance vector is unavailable")
    tokenizer = _tokenizer(receipt, candidate)
    if _fragment(candidate, tokenizer) != expected[0]:
        raise LoadError("live tokenizer does not match reviewed conformance vector")
    receipt.assert_live()
    return tokenizer
