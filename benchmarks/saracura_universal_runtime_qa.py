"""Offline, public-safe QA for the installed Saracura universal runtime.

This checkout-only command intentionally imports the historical training path
only for read-only parity checks against the two committed public examples. It
does not read accepted rows, holdout records, corpus roots, or any path inferred
from configuration. Output contains aggregate booleans and measurements only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from benchmarks import saracura_universal_training as training
from saracura.backends.saracura_universal import (
    SaracuraUniversalBackend,
    verify_training_capsule,
)
from saracura.contracts import parse_request_json
from saracura.universal.rendering import MAX_CONTEXT_TOKENS, MAX_CRITERION_TOKENS, render_choice

ROOT = Path(__file__).parents[1]
PUBLIC_EXAMPLES = (
    ROOT / "examples" / "ptbr-saracura-request.json",
    ROOT / "examples" / "en-saracura-request.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saracura-universal-runtime-qa")
    parser.add_argument("--encoder-snapshot", type=Path, required=True)
    parser.add_argument("--training-capsule", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "both"), required=True)
    return parser


def _request(path: Path) -> Any:
    return parse_request_json(path.read_bytes())


def _render(request: Any) -> Any:
    question = request.questions[0]
    return render_choice(
        locale=request.locale,
        domain=request.domain,
        instruction=question.instruction,
        state=request.state,
        criteria=question.criteria,
    )


def _ranked_ids(scores: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=scores.__getitem__, reverse=True))


def _versions() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in ("torch", "safetensors", "transformers", "tokenizers")
    }


def _source_commit_exists(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _runtime_values(backend: SaracuraUniversalBackend, request: Any) -> tuple[Any, Any, Any]:
    """Return private intermediates only to compare in memory; never emit them."""

    rendered = _render(request)
    tokens = backend._tokenize_joint(rendered)
    torch = backend._torch
    assert torch is not None and backend._encoder is not None and backend._ranker is not None
    with torch.inference_mode():
        output = backend._encoder(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], return_dict=True
        )
        embeddings = backend._mean_pool(output.last_hidden_state, tokens["attention_mask"])
        logits = backend._ranker.logits(embeddings[:1], embeddings[1:])
    return tokens, embeddings.detach().to("cpu"), logits.detach().to("cpu")


def _training_values(
    snapshot: Path, checkpoint: bytes, request: Any, torch: Any
) -> tuple[Any, Any, Any]:
    """Run only the historical public-text encoder/projection calculation."""

    loaded = training._load_verified_minilm(snapshot)
    rendered = _render(request)
    context_ids, context_masks, context_embeddings = training._extract_bounded_embeddings(
        loaded, [rendered.context], MAX_CONTEXT_TOKENS, torch.device("cpu"), torch
    )
    criterion_ids, criterion_masks, criterion_embeddings = training._extract_bounded_embeddings(
        loaded,
        list(rendered.criteria),
        MAX_CRITERION_TOKENS,
        torch.device("cpu"),
        torch,
    )
    model = training._model_from_checkpoint(checkpoint).eval()
    with torch.inference_mode():
        context = torch.nn.functional.normalize(model.project_context(context_embeddings), dim=-1)
        criteria = torch.nn.functional.normalize(
            model.project_criterion(criterion_embeddings), dim=-1
        )
        logits = (context @ criteria.T).reshape(-1) * model.log_scale.exp()
    return (
        (context_ids, context_masks, criterion_ids, criterion_masks),
        torch.cat((context_embeddings, criterion_embeddings), dim=0),
        logits,
    )


def _same_unpadded(runtime_tokens: Any, training_tokens: Any) -> bool:
    context_ids, context_masks, criterion_ids, criterion_masks = training_tokens
    ids, masks = runtime_tokens["input_ids"].cpu(), runtime_tokens["attention_mask"].cpu()
    expected_ids = [context_ids[0], *criterion_ids]
    expected_masks = [context_masks[0], *criterion_masks]
    return all(
        bool(
            (ids[index, : int(mask.sum().item())] == token_ids[: int(mask.sum().item())])
            .all()
            .item()
        )
        and bool(
            (masks[index, : int(mask.sum().item())] == mask[: int(mask.sum().item())]).all().item()
        )
        for index, (token_ids, mask) in enumerate(zip(expected_ids, expected_masks, strict=True))
    )


def _one_device(snapshot: Path, capsule: Path, device: str) -> dict[str, object]:
    import psutil  # type: ignore[import-untyped]
    import torch  # type: ignore[import-not-found]

    started = time.perf_counter()
    backend = SaracuraUniversalBackend(
        encoder_snapshot=snapshot,
        training_capsule=capsule,
        device=device,  # type: ignore[arg-type]
    )
    try:
        backend.prepare()
        cold_prepare_ms = (time.perf_counter() - started) * 1000
        results: list[tuple[bool, bool, bool, tuple[str, ...], dict[str, float]]] = []
        verified = verify_training_capsule(capsule)
        for example in PUBLIC_EXAMPLES:
            request = _request(example)
            runtime_tokens, runtime_embeddings, runtime_logits = _runtime_values(backend, request)
            training_tokens, training_embeddings, training_logits = _training_values(
                snapshot, verified.values["checkpoint.safetensors"], request, torch
            )
            rtol, atol = (1e-5, 1e-6) if device == "cpu" else (1e-4, 1e-5)
            embedding_ok = bool(
                torch.isclose(runtime_embeddings, training_embeddings, rtol=rtol, atol=atol)
                .all()
                .item()
            )
            logits_ok = bool(
                torch.isclose(runtime_logits, training_logits, rtol=rtol, atol=atol).all().item()
            )
            decision = backend.score_universal_choice(request, request.questions[0], b"")
            results.append(
                (
                    _same_unpadded(runtime_tokens, training_tokens),
                    embedding_ok,
                    logits_ok,
                    _ranked_ids(decision.raw_scores),
                    decision.raw_scores,
                )
            )
        for _ in range(3):
            for example in PUBLIC_EXAMPLES:
                request = _request(example)
                backend.score_universal_choice(request, request.questions[0], b"")
        samples: list[float] = []
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        for _ in range(20):
            for example in PUBLIC_EXAMPLES:
                request = _request(example)
                tick = time.perf_counter()
                backend.score_universal_choice(request, request.questions[0], b"")
                samples.append((time.perf_counter() - tick) * 1000)
                peak_rss = max(peak_rss, process.memory_info().rss)
        repeat_rankings: list[tuple[str, ...]] = []
        for example in PUBLIC_EXAMPLES:
            request = _request(example)
            repeat_rankings.append(
                _ranked_ids(
                    backend.score_universal_choice(request, request.questions[0], b"").raw_scores
                )
            )
        ordered = sorted(samples)
        return {
            "available": True,
            "cold_prepare_ms": round(cold_prepare_ms, 3),
            "warm_p50_ms": round(ordered[len(ordered) // 2], 3),
            "warm_p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
            "throughput_decisions_per_second": round((len(samples) / 2) / (sum(samples) / 1000), 3),
            "peak_rss_bytes": peak_rss,
            "input_shape": {"public_examples": 2, "questions": 1, "criteria": 4},
            "token_ids_and_masks_match_training": all(item[0] for item in results),
            "embeddings_match_training": all(item[1] for item in results),
            "logits_match_training": all(item[2] for item in results),
            "deterministic_rankings": [item[3] for item in results] == repeat_rankings,
            "rankings": [item[3] for item in results],
            "scores": [item[4] for item in results],
        }
    finally:
        backend.close()


def _same_scores(
    left: dict[str, object], right: dict[str, object], *, rtol: float, atol: float
) -> bool:
    left_scores = left["scores"]
    right_scores = right["scores"]
    if not isinstance(left_scores, list) or not isinstance(right_scores, list):
        return False
    return all(
        isinstance(first, dict)
        and isinstance(second, dict)
        and first.keys() == second.keys()
        and all(
            math.isclose(float(first[key]), float(second[key]), rel_tol=rtol, abs_tol=atol)
            for key in first
        )
        for first, second in zip(left_scores, right_scores, strict=True)
    )


def _public_result(result: dict[str, object]) -> dict[str, object]:
    public = dict(result)
    public.pop("scores")
    return public


def main(argv: list[str] | None = None) -> int:
    values = _parser().parse_args(argv)
    # This is checkout-only QA, not installed runtime behavior.  The
    # historical verifier independently rechecks the sealed evidence before
    # the installed verifier and public-text parity path run.
    training.verify_training_capsule(values.training_capsule)
    capsule = verify_training_capsule(values.training_capsule)
    if not _source_commit_exists(str(capsule.manifest["source_commit"])):
        raise RuntimeError("approved source commit is unavailable")
    if values.device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            print(json.dumps({"device": "mps", "available": False}, sort_keys=True))
            return 0
    if values.device != "both":
        result = _one_device(values.encoder_snapshot, values.training_capsule, values.device)
        payload: dict[str, object] = {"result": _public_result(result)}
    else:
        cpu_first = _one_device(values.encoder_snapshot, values.training_capsule, "cpu")
        cpu_second = _one_device(values.encoder_snapshot, values.training_capsule, "cpu")
        import torch

        if not torch.backends.mps.is_available():
            payload = {
                "cpu": _public_result(cpu_first),
                "cpu_repeat": _public_result(cpu_second),
                "cpu_deterministic": _same_scores(cpu_first, cpu_second, rtol=0.0, atol=0.0),
                "mps": {"available": False},
            }
        else:
            mps_first = _one_device(values.encoder_snapshot, values.training_capsule, "mps")
            mps_second = _one_device(values.encoder_snapshot, values.training_capsule, "mps")
            payload = {
                "cpu": _public_result(cpu_first),
                "cpu_repeat": _public_result(cpu_second),
                "mps": _public_result(mps_first),
                "mps_repeat": _public_result(mps_second),
                "cpu_deterministic": _same_scores(cpu_first, cpu_second, rtol=0.0, atol=0.0),
                "mps_deterministic": _same_scores(mps_first, mps_second, rtol=0.0, atol=0.0),
                "cross_device_logits": _same_scores(cpu_first, mps_first, rtol=1e-4, atol=1e-5),
                "cross_device_rankings": cpu_first["rankings"] == mps_first["rankings"],
            }
    print(
        json.dumps(
            {
                "device": values.device,
                "source_commit_present": True,
                "training_time_capsule_verifier": "passed",
                "installed_capsule_verifier": "passed",
                "dependencies": _versions(),
                **payload,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
