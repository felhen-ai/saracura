---
title: PT-BR typed decisions for lm-evaluation-harness
kind: reference
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: research/lm_eval/ptbr_typed_decisions/README.md
globalRef: qmd://saracura/research/lm_eval/ptbr_typed_decisions/README.md
reviewCadenceDays: 90
lastReviewedAt: 2026-10-10
sourceRefs:
  - https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench
  - https://github.com/EleutherAI/lm-evaluation-harness
related:
  - research/README.md
supersedes: []
supersededBy: []
sensitivity: public
---

# PT-BR typed decisions

Seven closed-question tasks on real Brazilian Portuguese text (offensive language and its target, fact-check
verdict, answer adequacy, scientific area, audit-court jurisprudence area, legislative theme), from openly licensed
sources with human labels. Dataset and card: https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench

Each item lists the options with letters and the model scores each letter (`multiple_choice`, loglikelihood).
The headline metric is balanced accuracy (mean per-class recall), because several tasks are imbalanced; `acc` is
reported too. Zero-shot by default.

## Tasks

- `ptbr_typed_decisions` (group, unweighted mean over the seven tasks)
- `ptbr_decisions_camara_tema`, `ptbr_decisions_factck_veracidade`, `ptbr_decisions_faquad_resposta`,
  `ptbr_decisions_juristcu_area`, `ptbr_decisions_olid_alvo`, `ptbr_decisions_olid_ofensivo`,
  `ptbr_decisions_scielo_area`

## Usage

```bash
lm_eval --model hf --model_args pretrained=<model> --include_path research/lm_eval/ptbr_typed_decisions --tasks ptbr_typed_decisions
```

## Differences from the card's protocol

The card's table scores typed-decision servers through `research/ptbr_bench.py`, with option order shuffled per
item. This harness version keeps the dataset's option order and uses letter loglikelihood, so its numbers are
comparable across language models run here, not directly with the card's decision-model table.

## Licenses

Compilation CC BY 4.0; each task keeps its source license (CC BY 4.0, MIT, public government data). Details and
citations on the dataset card.
