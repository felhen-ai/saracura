---
title: Chance-corrected scoring V2
kind: reference
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: research/scoring-v2.md
globalRef: qmd://saracura/research/scoring-v2.md
reviewCadenceDays: 60
lastReviewedAt: 2026-11-01
sourceRefs:
  - https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench
  - https://github.com/NandhaKishorM/laya
related:
  - research/README.md
  - research/ptbr_bench.py
  - research/validate_submission.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Scoring V2: chance-corrected PT-BR ranking

Effective 2026-11-01, `ptbr-chance-corrected-v2` is the principal ranking metric. This versions the **score**, not the dataset or inference protocol. The original 14 reports, raw balanced accuracy, raw bootstrap intervals, checkpoint revisions and 5,196 predictions per model remain unchanged. No new model runs were performed.

For each task, let BA be balanced accuracy (mean recall across its K gold classes). Compute:

```text
score_task = max(0, (BA - 1/K) / (1 - 1/K))
score_v2 = 100 * mean(score_task for each of the seven tasks)
```

Equal task weights are retained. Normalize each task before averaging; subtracting a single baseline from the aggregate is incorrect. A chance-level classifier scores zero in expectation before clipping; finite-sample results can differ. Scores at or below chance are clipped to zero per task; 100 means perfect classification. Clipping loses information about below-chance performance, so the JSON also retains signed adjusted values and raw BA. This convention follows the [Open LLM Leaderboard normalization approach](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization); chance correction is useful but is not universal across benchmarks. It does not make scores comparable across different datasets or protocols.

| Task | K | Chance BA |
|---|---:|---:|
| camara_tema | 24 | 1/24 |
| factck_veracidade | 3 | 1/3 |
| faquad_resposta | 2 | 1/2 |
| juristcu_area | 10 | 1/10 |
| olid_alvo | 3 | 1/3 |
| olid_ofensivo | 2 | 1/2 |
| scielo_area | 8 | 1/8 |

The V2 conversion uses exact per-item predictions, not rounded per-task summaries. Original published scores are retained as published, so tiny rounding differences may occur. No V2 confidence intervals have been computed. Original bootstrap intervals apply **only to original BA**; small differences in the V2 ordering do not establish statistically significant superiority.

The main ranking includes 12 external models. The two Saracura checkpoints, which were fine-tuned on these tasks' training splits, remain outside it. Unknown upstream training exposure for external checkpoints remains unknown.

## Reproduce without running models

Download the original report JSONs listed in `research/rounds/2026-10-21/final-ranking.json` from [the immutable completed round](https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/tree/11d2412c85441d56ff5bf914c647a7a56d18b75a/results/comparable-round-2026-10-21-complete). Obtain each `data/<task>/test.jsonl` at dataset revision `00f5d6dcc6679751723289889b8b0b28b71cbcb9`, saved locally as `<task>.test.jsonl`. Then run (standard library only):

```bash
python research/normalize_scores.py \
  --reports /path/to/completed-round-reports \
  --dataset-cache /path/to/frozen-test-splits \
  --output /path/to/derived-v2
```

The converter verifies all report SHA-256 values against the frozen manifest, all split hashes against the original reports, gold-label sequences, class counts, seven tasks, 5,196 items and zero failures. It emits `scoring-v2.json` and `scoring-v2.csv`. These are additive summaries; they never overwrite original reports. The original `final-ranking.json` records V1; the sibling `scoring-v2.json` records the new score and supports sorting by `normalized_0_100_points`.

New participants still submit the original evaluator's full validated JSON via the [existing guide](evaluation-guide.md). Maintainers derive V2 from its predictions. No new evaluation, training or labeling step is required by this score migration.
