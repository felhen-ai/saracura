---
title: Benchmark participation review and publication drafts
kind: report
area: research
project: saracura
collection: saracura
owner: saracura-maintainers
status: draft
canonical: docs/action/benchmark-participation-2026-10-31.md
globalRef: qmd://saracura/docs/action/benchmark-participation-2026-10-31.md
reviewCadenceDays: 90
lastReviewedAt: 2026-10-31
sourceRefs:
  - research/evaluation-guide.md
  - research/rounds/2026-10-21/final-ranking.json
  - https://github.com/huggingface/blog
  - https://huggingface.co/Mapika
  - https://huggingface.co/Abhinav-jigsawstack
  - https://huggingface.co/jaredpalmer
  - https://github.com/NandhaKishorM/laya
related:
  - .github/ISSUE_TEMPLATE/benchmark-submission.yml
supersedes: []
supersededBy: []
sensitivity: public
---

# Participation review and drafts

Drafts only: no outreach or announcement has been sent. No new model evaluation,
VM work, paid compute, private pilot or new submission system is part of this task.

## Small corrections to the existing path

- Visitors can open the ranking without cloning or installing anything.
- Evaluator examples pin the completed round's dataset revision and use
  `uv run --no-project` to avoid installing the repository's development packages.
- The historical default-context Laya CPU recipe is explicitly distinct from the
  final round's larger-context Laya configuration. Historical evidence is preserved.
- Bridge report names include `constrained-letter`, with a matching validation
  command. The evaluator and runtime are unchanged.
- Verification: both standard-library CLIs start with `--no-project`; the existing
  full Laya report passes `validate_submission.py` (seven tasks, recomputed mean
  balanced accuracy 39.2%; warning: checkpoint subfolder is unknown). No inference
  or fresh model evaluation was run.
- The existing issue form asks for benchmark-task/source-dataset training history
  (yes/no/unknown and a source). Known task-tuned results are displayed separately;
  an unknown history does not certify unseen data. Private tracks are not required.

## Public Space correction

The existing “Enviar um modelo” tab now links both submission routes, mentions
LLMs through the local bridge, requests training-history disclosure and explicitly
says that historical private tracks are not required. Space commit:
`4b842dc6fe62c25d4feb4f124e4db1996d1b0f84`.
Anonymous readback matches the published page. The original `results.json` is
byte-identical (SHA256 `07158ba401b310524d3b482588cdf66ca7e5b2dcdb5b69ef6c45f11d74bb7c1f`).
All fourteen reports and model scores are preserved. This is a correction to the
existing participation instructions, not an outreach post.

## Distribution order

1. Publish one Hugging Face community article linking the existing Space, frozen
   reports and guide. The Hugging Face blog contribution instructions explicitly
   direct unaffiliated external articles to the community blog. This is an existing
   discovery channel, not an endorsement or guaranteed distribution.
2. Reuse a short Portuguese announcement on LinkedIn. No new website, newsletter
   or advertising campaign is needed.
3. After the public announcement, send at most one relevant technical invitation
   to each selected author. Ask for corrections to our existing evidence first;
   do not ask them to train a model or make a large commitment.

Do not lead with Reddit: r/LocalLLaMA's current rules restrict predominantly
LLM-written material and self-promotion. That channel would require additional
human authorship/account-history review and is unnecessary for this first step.
Source: https://www.reddit.com/r/LocalLLaMA/about/rules.json

A useful first outcome is one substantive external configuration correction or
one valid external submission, rather than more followers or download counters.
Use existing discussions/issues to observe it; add no tracker or automation.

## Public article draft — Hugging Face

**Title: A reproducible PT-BR typed-decision benchmark: 12 external models, open reports**

We maintain the PT-BR typed-decisions benchmark and have published a completed
comparison of 12 external models on 5,196 test items across seven Portuguese
classification tasks. The Space includes model revisions, per-task results and
links to the underlying reports.

The benchmark measures selecting a typed answer from supplied options. It is not
a general ranking of Portuguese chat quality or evidence of performance in a
particular production workflow. The headline metric is mean balanced accuracy
across tasks; option order is shuffled. Native decision runtimes and general
LLMs using constrained-letter readout are identified in the published evidence.
Hardware and runtime settings differ, so the execution times are not a controlled
speed comparison.

We also developed Saracura. Its two checkpoints were tuned on the benchmark tasks,
so we exclude them from the main comparison and show their results separately
with that disclosure. For the other models, absence of identified task-specific
tuning does not certify that their upstream training data excluded the tasks.

All 14 full reports remain available: 5,196 items, seven tasks and zero request
failures per report, checked with the public submission validator. Validation
checks consistency with the public data; it does not establish unseen-data
performance or prove a checkpoint's identity by itself.

If you maintain one of the listed models, a correction to its documented runtime
or configuration would already be useful. If you want to submit another public
model, the existing guide covers the local server or LLM bridge, a small smoke
check, the full run and validation. Private data or manual labeling is not required.
Submissions are reviewed before inclusion; nothing is listed automatically.

- Ranking: https://huggingface.co/spaces/felhen-ai/ptbr-typed-decisions-leaderboard
- Frozen reports: https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/tree/11d2412c85441d56ff5bf914c647a7a56d18b75a/results/comparable-round-2026-10-21-complete
- Guide: https://github.com/felhen-ai/saracura/blob/main/research/evaluation-guide.md

## Short Portuguese announcement — LinkedIn

Publicamos uma comparação aberta de 12 modelos externos em sete tarefas de
classificação em português, com 5.196 exemplos, revisões fixadas e relatórios
consultáveis.

Eu participo da manutenção do benchmark e do Saracura. Os nossos dois modelos
foram ajustados nessas tarefas e aparecem separados da classificação principal.
A comparação mede decisões entre alternativas; não pretende dizer qual é o melhor
modelo de português para qualquer aplicação.

O ranking e o guia para avaliar outro modelo já estão disponíveis. Comentários
sobre as configurações dos modelos e contribuições com novos relatórios são
bem-vindos — sem necessidade de preparar dados privados ou rotular textos.

https://huggingface.co/spaces/felhen-ai/ptbr-typed-decisions-leaderboard

## Four selected authors and individual drafts

These identities and project relationships were checked on their public profiles
on 2026-10-31. The proposed channel is the relevant model's existing Hugging Face
Community page, subject to rereading current discussions immediately before any
approved message. Do not open unrelated GitHub bug reports for promotion.
The scores below refer to the pinned completed round, not necessarily the latest
checkpoint or a statistically significant difference between models.

### Mark Marosi / Mapika — Decider

Profile: https://huggingface.co/Mapika
Existing thread (follow up here; do not open another):
https://huggingface.co/Mapika/decider-4b/discussions/4

Our earlier message reports 64.1% under the older evaluation. The follow-up
updates it to the frozen comparable round rather than treating it as a new contact.

Hi Mark — an update to our earlier post: the completed comparable round gives
Decider-4B 63.56% mean balanced accuracy on the seven public PT-BR tasks
(5,196 items). This supersedes the 64.1% figure from our older evaluation; the
frozen configuration and full report are linked in the ranking below. We now
show our task-tuned Saracura checkpoints separately from the main comparison.
Would you flag any issue in Decider's documented runtime? Feedback on the
existing evidence is enough; no new run is needed.
https://huggingface.co/spaces/felhen-ai/ptbr-typed-decisions-leaderboard

### Abhinav / Interfaze — Lev

Profile: https://huggingface.co/Abhinav-jigsawstack
Model/community: https://huggingface.co/interfaze-ai/lev/discussions

Hi Abhinav — we published a pinned Lev run on seven PT-BR typed-decision tasks:
63.09% mean balanced accuracy over 5,196 items. Could you check whether the
published runtime and preprocessing configuration represents Lev appropriately?
The full report and evidence are linked below. I maintain this benchmark and
Saracura; our task-tuned models are shown separately. A configuration correction
alone would be useful; there is no need to run another evaluation.

https://huggingface.co/spaces/felhen-ai/ptbr-typed-decisions-leaderboard

### Jared Palmer — Kev

Profile: https://huggingface.co/jaredpalmer
Model/community: https://huggingface.co/jaredpalmer/kev-4b/discussions

Hi Jared — we evaluated pinned Kev-4B and Kev-0.8B checkpoints on seven PT-BR
typed-decision tasks, obtaining 62.11% and 50.83% mean balanced accuracy over
5,196 items each. Could you flag any issue in the recorded runtime/readout
configuration? Reports are linked below. We maintain Saracura too, and have
separated our task-tuned checkpoints from the main comparison. This is a request
for feedback on existing evidence, not a request to train or rerun the models.

https://huggingface.co/spaces/felhen-ai/ptbr-typed-decisions-leaderboard

### Laya maintainers / NandhaKishorM

Project: https://github.com/NandhaKishorM/laya
Model/community: https://huggingface.co/convaiinnovations/laya/discussions

Hi Laya team — we published a pinned multilingual Laya run on seven PT-BR
typed-decision tasks: 39.18% mean balanced accuracy over 5,196 items. We document
the native 48-token option cap and larger-context configuration in the evidence.
Could you flag any mismatch with the intended runtime behavior? I maintain this
benchmark and the Laya-derived Saracura-v0; our task-tuned models are separated
from the main comparison. Feedback on the existing configuration would be enough.

https://huggingface.co/spaces/felhen-ai/ptbr-typed-decisions-leaderboard
