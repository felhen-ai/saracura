---
title: Saracura research scripts
kind: reference
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: research/README.md
globalRef: qmd://saracura/research/README.md
reviewCadenceDays: 60
lastReviewedAt: 2026-10-08
sourceRefs:
  - https://github.com/NandhaKishorM/laya
  - https://github.com/jaredpalmer/kev
related:
  - README.md
  - docs/decisions/0005-open-base-benchmark-first.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Research scripts

Plain scripts, one concern each, run in the order below. They are the whole method behind the published benchmark and
checkpoints (ADR 0005). They are not part of the `saracura` package and are excluded from the strict typing lane;
`ruff` still runs on them.

Expected layout, relative to the directory the scripts run from: `data/` (benchmark splits and generated sets),
`items/` (tokenized training items), `runs/` (checkpoints), `results/` (JSON reports), `models/laya/multilingual`
(the upstream base checkpoint). Internal data used in our runs (marketplace listings, Felhen documents) is not in the
repository; the scripts that read it (`segmento.py`, `combo.py`, `agree_qwen.py`) will not run without it.

## Benchmark

| Script | What it does |
|---|---|
| `build_bench.py` | Builds the first benchmark version from the Hugging Face sources (OLID-BR, FACTCK.BR, FaQuAD-NLI, SciELO, JurisTCU, Câmara). |
| `build_bench_v1.py` | Current version: larger train splits, a dev split, class descriptions (`data/bench/descriptions.json`), extra legislative data from LegiSubject-Br. |
| `bench.py` | `baselines` (majority, TF-IDF + logistic regression per task), `eval` (any Laya-compatible checkpoint, balanced accuracy, option order shuffled per item), `prep` (tokenized training items). Configured by `BENCH_DIR`, `BENCH_MAX_CHARS`, `BENCH_MAX_LEN`, `BENCH_HEAD_MAX_LEN`, `BENCH_TAG`. |
| `faq_bacen.py` | The held-out transfer task: Central Bank FAQ answer matching, rebuilt exactly as the earlier harness planned it (lexical baseline 265/373). |
| `bench_llm.py` | Zero-shot reference with a local open-weight LLM served by vLLM, and generation of one-sentence class descriptions. |

## Question generation (teachers)

| Script | What it does |
|---|---|
| `gen_questions.py` | Local teacher (vLLM): 3 to 5 varied questions per real text, answered 5 times with temperature; the answer distribution is the target. Also `dedupe` (removes training questions whose instruction appears in the held-out set) and `prep`. |
| `gen_openrouter.py` | Same idea with a large open-weight teacher over OpenRouter (`qwen/qwen3.5-397b-a17b`, Apache-2.0), two calls per text, budget ceiling in dollars. Disable the model's reasoning or it burns the output budget. |
| `gen_usecases.py` | Synthetic texts for eight product use cases, written from a sampled brief (persona, tone, size, target labels), then answered blind; texts whose blind answers disagree with the brief are dropped. |
| `eval_gen.py` | Agreement with the teacher on held-out generated questions, by question type and source. |

## Training

| Script | What it does |
|---|---|
| `laya_finetune_mps.py` | The upstream Laya fine-tuning script, unchanged (Apache-2.0, Convai Innovations). RLCD loss, per-type temperature calibration, checkpoint export. |
| `run_ft_cuda.py` | Runs the upstream script on CUDA (it only offers MPS and CPU) and lets `LR_SCALE` scale the fixed learning rates. |
| `prep_items.py`, `combo.py` | Item preparation for the marketplace task and for the combined training set (benchmark + translated typed-decisions + listings). |
| `eval_eikos.py` | Runs another open decision model (Eikos, `caiovicentino1/Eikos-4B`) on the benchmark with its own published inference code and the same protocol. |
| `kev_data.py`, `kev_bench_long.py`, `kev_score.py` | Conversion of the same data to the Kev request format, benchmark with the 2048-token training context, balanced accuracy from Kev predictions. |
| `run_*.sh` | The exact chains used for each training round, as run on one RTX 5090 (train with `--micro-batch 4 --grad-accum 16` and gradient checkpointing for 2048-token contexts, or it runs out of 32 GB). |

## Comparison table of open decision models

| Script | What it does |
|---|---|
| `eval_http.py` | Runs any System One-compatible server (`POST /v1/systemone`) on the benchmark and on the Central Bank FAQ task, same protocol as `bench.py` (shuffled option order, `BENCH_MAX_CHARS=3500`). Accepts `noul` answers as `noul` or `probability`. |
| `board_pyserve.py` | Minimal `/v1/systemone` wrapper for models that only ship a Python class (Intern-Decision, Jet). |
| `llm_bacen.py` | Central Bank FAQ task with a local open LLM (Qwen 3.8 27B over vLLM), answers restricted to the four letters; `bench_llm.py ceiling` is the benchmark counterpart. |

Each model is served by its author's own server and package, one at a time, on one RTX 5090. The table and the
per-model results are on the benchmark card.

## Adaptation track

How much a decision model improves with a few labeled examples of the target task, and how that compares with a
classical baseline trained on the same examples.

| Script | What it does |
|---|---|
| `adapt_build.py` | Draws N training examples per task (fixed seed, nested sizes) and writes them as `kev.train` requests. |
| `adapt_tfidf.py` | TF-IDF + logistic regression trained on the same N examples (50, 100, 400 and the full split). |
| `ci.py` | 95% confidence intervals by item bootstrap, per task and for the mean, from results with per-item answers. |

The decision-model points start from `jaredpalmer/kev-4b`, which has seen none of the benchmark data: one run per
size (`kev.train --init_from jaredpalmer/kev-4b --epochs 2 --lr 2e-5`, examples repeated 4 times for 50 and 100 and
twice for 400), each served with `kev.serve` and measured with `eval_http.py`, which also saves per-item answers.

## Publication

`publish.py` uploads the benchmark and a checkpoint to the Hub (token on stdin, never written to disk); `verify_hub.py`
loads the published checkpoint back and runs a prediction.

## Reporting rules

- Balanced accuracy (mean per-class recall), option order shuffled per item, up to 1,000 test items per task.
- Teacher agreement is reported as agreement, never as accuracy.
- Any checkpoint evaluated with a context different from its training context gets worse numbers; pass the same
  `max_len` and `head_max_len`.
