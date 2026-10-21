---
title: Evaluate and submit a model on the PT-BR benchmark
kind: guide
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: research/evaluation-guide.md
globalRef: qmd://saracura/research/evaluation-guide.md
reviewCadenceDays: 60
lastReviewedAt: 2026-10-21
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

# Evaluate and submit a model on the PT-BR benchmark

English first, Português do Brasil [below](#português-do-brasil).

Prerequisites / pré-requisitos: Git, curl and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Start from a fresh directory / comece em uma pasta nova:

```bash
git clone https://github.com/felhen-ai/saracura.git
cd saracura
```

## English

For a chat/instruct LLM, start with the [local bridge](#chat-llms-through-the-local-bridge), then run the smoke, full evaluation and validation below. If your model already serves System One, start at step 1.

Two scripts, both Python standard library only (no ML packages needed to run them):

- `research/ptbr_bench.py` sends the 5,196 test items of the seven public tasks to a System One-compatible server
  (`POST /v1/systemone`) and writes a report with per-item answers.
- `research/validate_submission.py` re-downloads the public data, recomputes every score from your per-item answers and
  tells you whether the report can be submitted.

### 1. Serve a model

Any server that answers `POST /v1/systemone` works; keep it on `127.0.0.1`. The model is served by its author's own
package, one at a time. A CPU-friendly example is the upstream Laya multilingual checkpoint (322M parameters, Apache-2.0,
Convai Innovations), the base of `felhen-ai/saracura-ptbr-v0`.

Settings that matter, verified in a smoke run (14 requests, 2 per task, 0 failures):

| Step | Value |
|---|---|
| Python | 3.12; `laya[serve]==0.3.20`, in the separate environment below |
| Weights | `huggingface_hub.snapshot_download(repo_id="convaiinnovations/laya", revision="7b928d828b7b0e022f929d9bd2e44165aa270148", allow_patterns=["multilingual/*"], token=False)` |
| Server process | separate process, `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false`, `torch.set_num_threads(4)` |
| Router | `Router(models={"multilingual": <snapshot>/multilingual}, device="cpu", default="multilingual", auto_task_detection=False, max_loaded=1)`, then `router.preload(["multilingual"])`, then `uvicorn.run(create_app(router=router), host="127.0.0.1", port=18021)` |

Force the multilingual model. With the bundled `convaiinnovations/laya` id the upstream router picks a model by detected
language, which is not what this benchmark measures. The request alias is `convaiinnovations/laya-multilingual`; the
checkpoint identity is `convaiinnovations/laya` at the commit above, subfolder `multilingual`.

The complete CPU run finished on 2026-10-21: 5,196 items, seven tasks, zero failed requests. This checks the evaluation
path; it does not replace the historical leaderboard measurements. The following commands use the upstream server.
Keep this terminal running; run the evaluator from a second terminal in the same repository directory.
[Execution evidence, full report and frozen evaluator](https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/tree/main/results/contribution-check-2026-10-21)
record the exact model, data, runtime and protocol. Elapsed times are operational, not isolated speed benchmarks.

### Copyable CPU example / exemplo de CPU pronto para copiar

Create a separate ML environment outside the repository's default development environment:
crie um ambiente de ML separado do ambiente padrão do repositório.

```bash
uv venv --python 3.12 "$HOME/.cache/ptbr-benchmark-laya/venv"
uv pip install --python "$HOME/.cache/ptbr-benchmark-laya/venv/bin/python" 'laya[serve]==0.3.20' 'torch==2.14.1' 'transformers==5.20.2'
```

Download the pinned public checkpoint explicitly once / baixe explicitamente o checkpoint público fixado:

```bash
"$HOME/.cache/ptbr-benchmark-laya/venv/bin/python" - <<'PYTHON'
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="convaiinnovations/laya",
    revision="7b928d828b7b0e022f929d9bd2e44165aa270148",
    allow_patterns=["multilingual/*"],
    local_dir=str(Path.home() / ".cache/ptbr-benchmark-laya/model"),
    token=False,
)
PYTHON
```

Start the upstream server with the local snapshot and networking disabled for the Hub:
inicie o servidor original com os pesos locais e sem acesso ao Hub durante a inferência.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false \
  "$HOME/.cache/ptbr-benchmark-laya/venv/bin/python" - <<'PYTHON'
from pathlib import Path
import torch
import uvicorn
from laya import Router
from laya.serve import create_app

torch.set_num_threads(4)
model_dir = Path.home() / ".cache/ptbr-benchmark-laya/model/multilingual"
router = Router(
    models={"multilingual": str(model_dir)}, device="cpu", default="multilingual",
    auto_task_detection=False, max_loaded=1,
)
router.preload(["multilingual"])
uvicorn.run(create_app(router=router), host="127.0.0.1", port=18021)
PYTHON
```

The first step installs the optional ML packages; the evaluator and validator need only Python's standard library.
O primeiro passo instala os pacotes opcionais de ML; o avaliador e o validador precisam só da biblioteca padrão do Python.

### 2. Smoke run, then the full run

Run with a Python that has working certificates; `uv run --python 3.12` is the tested choice. Never disable TLS
verification.

```bash
# smoke: 2 items per task; the report says smoke and can never be submitted
uv run --python 3.12 research/ptbr_bench.py --base-url http://127.0.0.1:18021 \
  --model convaiinnovations/laya-multilingual --name laya-smoke --limit 2 \
  --output /tmp/laya-smoke.json \
  --revision 5450e73e634403993d9ac86d53d60a9b0ff76673

# full: all 5,196 items
uv run --python 3.12 research/ptbr_bench.py --base-url http://127.0.0.1:18021 \
  --model convaiinnovations/laya-multilingual --name laya-cpu-full \
  --output ./laya-cpu-full.json \
  --revision 5450e73e634403993d9ac86d53d60a9b0ff76673 \
  --checkpoint-id convaiinnovations/laya \
  --checkpoint-revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --subfolder multilingual --context "max_len 1024, head 256" \
  --quantization "none; CPU float32" --hardware "<cpu, memory>" \
  --server laya --server-version 0.3.20
```

- `--model` is the server alias or public identifier (for example `org/model`), never an endpoint URL; URLs are
  rejected before any work. `--model` é o alias do servidor ou um identificador público, nunca uma URL de endpoint.
- `--name` is a plain label. `--output PATH` chooses the file; an existing file is never overwritten, and the check
  happens before any request.
- `--checkpoint-revision` must be a 40-hex Hub commit or a `sha256:<64 hex>` digest of the weights. `main` or `latest`
  work for a trial but the report is then marked not submission-ready.
- Everything you do not declare is recorded as `unknown`. Nothing is guessed. Replace the hardware example with your actual machine; update quantization and context if you change the serving configuration.
- The report holds the dataset commit, protocol, evaluator hash, your declared metadata, timing and per-item answers.
  It does not hold the server address, prompts or credentials. `--base-url` is used only to send requests.
- Option order is shuffled with one random stream across tasks, so a `--limit` run does not shuffle the same way as a
  full run. Smoke numbers are not comparable with full numbers.

### 3. Read the output

Each line is `task  balanced accuracy  n  failed`. Balanced accuracy is the mean per-class recall; the mean over the
seven tasks has a 95% bootstrap interval. A failed item (server error, timeout, malformed or invalid answer) counts as
wrong and is listed by code (`transport`, `bad_response`, `invalid_choice`, ...). The final line says whether the run is
submission-ready. A run with any failure, any `--limit`, or a checkpoint without an immutable revision is not.

Certificate errors: use `uv run --python 3.12`, or set `SSL_CERT_FILE` to your CA bundle. A server that does not answer
shows up as `transport` failures on every item; fix the server, then run again with a new `--output`.

### 4. Validate and submit

```bash
uv run --python 3.12 research/validate_submission.py laya-cpu-full.json
```

It downloads the seven splits again at the dataset commit named in the report, compares gold labels, item counts and
predictions with that data, recomputes all scores and the interval, and lists unknown metadata as warnings. It writes
nothing and runs nothing from the report. Reports from older versions of `ptbr_bench.py` are rejected: run again.

Passing means the file is consistent with the public data and your declaration. It is not proof that the declared
checkpoint produced the answers, so maintainers reproduce a result before listing it.

Submit by opening a discussion on the benchmark dataset page
(<https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/discussions>) with the report attached, or use
the [GitHub issue template](https://github.com/felhen-ai/saracura/issues/new?template=benchmark-submission.yml). Maintainers review each submission
by hand; nothing is posted or accepted automatically.

Before you attach anything: the report must not contain private text, tokens, hostnames or paths. Do not paste
credentials or private prompts into the discussion. Only public models, without tuning on the private tracks, are
listed.

### Public and private tracks, and lm-eval

- The seven public tasks above are what `ptbr_bench.py` measures and what the public table lists.
- The private tracks (real-use e-mail triage and public comments, the fresh `camara_tema` test; definition in
  `research/private_tasks.json`) keep their texts and labels private. They are scored by the maintainers by manual
  review, never through this script.
- `research/lm_eval/ptbr_typed_decisions` is a separate protocol for `lm-evaluation-harness` (generic prompting of
  language models). Its numbers are not comparable with `ptbr_bench.py` numbers and are never mixed in one table.

#Native runtime preprocessing is part of the reported configuration. Laya 0.3.20 clips each option to 48 tokens internally; this affects 599 items in the frozen round. Declare that cap in evidence and context metadata, preserve original descriptions, and verify the whole state fits the selected runtime context. Earlier preflight holds over this cap remain historical. The evaluator and bridge add no truncation beyond the common 3,500-character state limit.

## Chat LLMs through the local bridge

General chat models have no System One server of their own. `research/llm_bridge.py` serves one pinned Transformers
snapshot on `127.0.0.1` with the same `POST /v1/systemone` contract (standard library core; torch and transformers are
imported only when you start it). Each question becomes one user chat message (state and question verbatim, options as
`A)` to `Z)` in the order received). Boolean questions use `A) Não` / `B) Sim` and return a discrete 0 or 1, not a
calibrated probability.

There are two separate, preregistered readout modes. `strict-generation` is the default format-following diagnostic:
greedy generation has a 32-token budget and only one exact uppercase letter is accepted. `constrained-letter` is the
common planned decision comparison for all seven general LLMs, regardless of a strict smoke result: it makes one causal
next-token forward and selects the largest finite float32 logit only among the valid option-letter token IDs. The first
legal option wins an exact tie. It has a one-token context budget. Every legal letter must be one distinct token that
round-trips exactly; otherwise startup/request validation fails before inference. This is a discrete restricted choice,
not a calibrated probability, and neither mode repairs, retries, or uses gold labels.

Use a separate environment and replace `ORG/NAME` and `<40-hex commit>` with the model and revision you decided on
beforehand (pinned versions as in the Laya example above):

```bash
uv venv --python 3.12 "$HOME/.cache/ptbr-bridge/venv"
uv pip install --python "$HOME/.cache/ptbr-bridge/venv/bin/python" 'torch==2.14.1' 'transformers==5.20.2' 'huggingface_hub'

# explicit, one-off download of the pinned snapshot into the normal Hub cache (the bridge never downloads)
"$HOME/.cache/ptbr-bridge/venv/bin/python" - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ORG/NAME", revision="<40-hex commit>", token=False)
PYTHON

# start: offline, loopback only, evidence written once (an existing file is never overwritten)
"$HOME/.cache/ptbr-bridge/venv/bin/python" research/llm_bridge.py \
  --model ORG/NAME --revision "<40-hex commit>" --device cpu --threads 4 --context 8192 \
  --chat-template-kwargs '{"enable_thinking": false}' --generation-timeout 90 \
  --readout constrained-letter --port 18031 --evidence-output ./bridge-evidence.json
```

- `--chat-template-kwargs` is mandatory: pass the model's non-thinking switch (or `{}` when it has none, after
  inspecting its template). Templates that call `strftime_now` are rendered with the date frozen to `2026-10-21`
  (`--template-date`); if the installed transformers cannot freeze it, startup fails. The template itself is the
  official `tokenizer.chat_template`, never replaced; its SHA256 is in the evidence. The accepted kwargs are a closed
  subset: `enable_thinking` (boolean), `thinking` (boolean) and `date_string` (valid `YYYY-MM-DD`); other keys, other
  types, `NaN`/`Infinity` and keys that collide after NFC are rejected before the model loads. The evidence file is hashed
  as `stable-json-v1` (sorted keys, compact, UTF-8, finite numbers): a stable file hash, not an RFC 8785 typed identity.
- `--device` is explicit (`cpu`, `mps`, `cuda`); the dtype follows the device (`float32`, `float16`, `bfloat16`) unless
  `--dtype` says otherwise, and is recorded. Attention is `eager`. `--auto-class AutoModelForImageTextToText` selects the
  conditional-generation class for text-only Qwen3.5; no `trust_remote_code` anywhere.
- The deadline (`--generation-timeout`, default 90 s) covers the whole request and is checked between decoding steps
  and between questions, so a single prefill cannot be interrupted. Keep the evaluator `--timeout` larger (default 120).
  Past the deadline, or on an invalid reply, the answer is HTTP 200 with `{}` for the question: the evaluator records
  `no_answer` once and does not retry. One request runs at a time; a concurrent one gets 429.
- `GET /health` returns the execution manifest (`evidence`: bridge SHA256, model and tokenizer revision, package
  versions, dtype, device, threads, context, selected output budget, template kwargs, template SHA256, prompt version
  and hash; readout, constrained-letter IDs/raw token strings and the `logits_to_keep` forward path when applicable;
  `evidence_sha256`) and the mutable `counters` (`invalid_output`, `deadline`, `overbudget`). No
  request text, endpoint or local path appears in either.

Smoke, then the full run, with `--server-version` set to the SHA256 of the exact evidence bytes. Capture `/health` before
and after the full run and compare `evidence_sha256` and the counters; any nonzero counter makes the run failed and
unranked.

```bash
EVIDENCE_SHA="$(shasum -a 256 ./bridge-evidence.json | cut -d' ' -f1)"
curl -s http://127.0.0.1:18031/health > health-before.json
uv run --python 3.12 research/ptbr_bench.py --base-url http://127.0.0.1:18031 --model ORG/NAME \
  --name llm-smoke --limit 2 --output /tmp/llm-smoke.json --revision 5450e73e634403993d9ac86d53d60a9b0ff76673
uv run --python 3.12 research/ptbr_bench.py --base-url http://127.0.0.1:18031 --model ORG/NAME \
  --name llm-full --output ./llm-full.json --revision 5450e73e634403993d9ac86d53d60a9b0ff76673 \
  --checkpoint-id ORG/NAME --checkpoint-revision "<40-hex commit>" \
  --context "8192 tokens; chat template; constrained-letter; 1 output token" --quantization "none; CPU float32" \
  --hardware "<cpu, memory>" --server ptbr-llm-bridge-constrained-letter --server-version "$EVIDENCE_SHA"
curl -s http://127.0.0.1:18031/health > health-after.json
```

Keep a sidecar next to the report that links the report SHA256, the evidence SHA256 and the source commits, with the
actual start arguments and private values redacted. Its readable metadata must use
`ptbr-llm-bridge-constrained-letter` or `ptbr-llm-bridge-strict-generation` exactly, and the report name must include
the same mode. Strict full runs are separate rows; native Q4 variants are separate rows too. Capture health before and
after a full run: any nonzero failure counter means the full run is failed and unranked. Numbers from the bridge follow
this protocol only; they are not mixed with `lm-evaluation-harness` numbers.

## Português do Brasil

Para um LLM de chat/instruções, comece pela [ponte local](#llms-de-chat-pela-ponte-local), depois faça o teste rápido, a avaliação completa e a validação abaixo. Se seu modelo já atende System One, comece no passo 1.

Dois scripts, só com a biblioteca padrão do Python (não precisam de pacote de ML):

- `research/ptbr_bench.py` envia os 5.196 itens de teste das sete tarefas públicas a um servidor compatível com o
  System One (`POST /v1/systemone`) e grava um relatório com as respostas item a item.
- `research/validate_submission.py` baixa de novo os dados públicos, recalcula cada nota a partir das suas respostas e
  diz se o relatório pode ser enviado.

### 1. Sirva um modelo

Serve qualquer servidor que responda `POST /v1/systemone`; deixe-o em `127.0.0.1`. Cada modelo roda no servidor do
próprio autor, um de cada vez. Um exemplo leve para CPU é o checkpoint multilíngue do Laya (322M de parâmetros,
Apache-2.0, Convai Innovations), a base do `felhen-ai/saracura-ptbr-v0`.

Configuração verificada em um teste rápido (14 requisições, 2 por tarefa, 0 falhas):

| Passo | Valor |
|---|---|
| Python | 3.12; `laya[serve]==0.3.20`, no ambiente separado da receita acima |
| Pesos | `huggingface_hub.snapshot_download(repo_id="convaiinnovations/laya", revision="7b928d828b7b0e022f929d9bd2e44165aa270148", allow_patterns=["multilingual/*"], token=False)` |
| Processo do servidor | processo separado, `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false`, `torch.set_num_threads(4)` |
| Router | `Router(models={"multilingual": <snapshot>/multilingual}, device="cpu", default="multilingual", auto_task_detection=False, max_loaded=1)`, depois `router.preload(["multilingual"])`, depois `uvicorn.run(create_app(router=router), host="127.0.0.1", port=18021)` |

Force o modelo multilíngue. Com o id `convaiinnovations/laya` do pacote, o router do projeto original escolhe o modelo
pelo idioma detectado, e não é isso que este benchmark mede. O alias da requisição é
`convaiinnovations/laya-multilingual`; a identidade do checkpoint é `convaiinnovations/laya` no commit acima, subpasta
`multilingual`.

A execução completa em CPU terminou em 21/10/2026: 5.196 itens, sete tarefas, nenhuma requisição com falha. Ela verifica
o percurso de avaliação e não substitui as medições históricas do ranking. Os comandos prontos estão na
[receita de CPU acima](#copyable-cpu-example--exemplo-de-cpu-pronto-para-copiar), com explicações nos dois idiomas.
Deixe o servidor aberto nesse terminal e execute os próximos comandos em outro terminal, na pasta do repositório.
As [evidências da execução](https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/tree/main/results/contribution-check-2026-10-21)
incluem relatório completo, avaliador executado e condições do teste. Os tempos são operacionais, não benchmarks de velocidade.

### 2. Teste rápido e depois a execução completa

Use um Python com certificados funcionando; `uv run --python 3.12` é a opção testada. Não desligue a verificação TLS.

```bash
# teste rápido: 2 itens por tarefa; o relatório diz smoke e nunca pode ser enviado
uv run --python 3.12 research/ptbr_bench.py --base-url http://127.0.0.1:18021 \
  --model convaiinnovations/laya-multilingual --name laya-smoke --limit 2 \
  --output /tmp/laya-smoke.json \
  --revision 5450e73e634403993d9ac86d53d60a9b0ff76673

# completo: os 5.196 itens (mesmos parâmetros da seção em inglês)
uv run --python 3.12 research/ptbr_bench.py --base-url http://127.0.0.1:18021 \
  --model convaiinnovations/laya-multilingual --name laya-cpu-full \
  --output ./laya-cpu-full.json \
  --revision 5450e73e634403993d9ac86d53d60a9b0ff76673 \
  --checkpoint-id convaiinnovations/laya \
  --checkpoint-revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --subfolder multilingual --context "max_len 1024, head 256" \
  --quantization "none; CPU float32" --hardware "<cpu, memória>" \
  --server laya --server-version 0.3.20
```

- `--name` é só um rótulo. `--output CAMINHO` escolhe o arquivo; arquivo existente nunca é sobrescrito, e a checagem
  vem antes de qualquer requisição.
- `--checkpoint-revision` precisa ser um commit de 40 hex do Hub ou um digest `sha256:<64 hex>` dos pesos. `main` ou
  `latest` servem para experimento, mas o relatório fica marcado como não enviável.
- O que você não declara fica como `unknown`. Nada é inventado. Troque o exemplo de hardware pela sua máquina real; atualize quantização e contexto se mudar a configuração do servidor.
- O relatório guarda o commit do dataset, o protocolo, o hash do avaliador, os metadados declarados, o tempo e as
  respostas item a item. Não guarda endereço do servidor, prompts nem credenciais.
- A ordem das opções é embaralhada com um único fluxo aleatório entre as tarefas, então uma execução com `--limit` não
  embaralha como a completa. Números do teste rápido não são comparáveis com os da execução completa.

### 3. Leia a saída

Cada linha é `tarefa  acurácia balanceada  n  failed`. Acurácia balanceada é a média da revocação por classe; a média
das sete tarefas vem com intervalo de 95% por bootstrap. Item com falha (erro do servidor, timeout, resposta malformada
ou inválida) conta como erro e aparece por código (`transport`, `bad_response`, `invalid_choice`, ...). A última linha
diz se a execução é enviável. Qualquer falha, qualquer `--limit` ou checkpoint sem revisão imutável a torna não enviável.

Erro de certificado: use `uv run --python 3.12` ou aponte `SSL_CERT_FILE` para o seu pacote de CAs. Servidor que não
responde aparece como falhas `transport` em todos os itens; corrija o servidor e rode de novo com outro `--output`.

### 4. Valide e envie

```bash
uv run --python 3.12 research/validate_submission.py laya-cpu-full.json
```

Ele baixa de novo as sete partições no commit do dataset indicado no relatório, compara rótulos, contagens e predições
com esses dados, recalcula todas as notas e o intervalo e lista metadados desconhecidos como avisos. Não grava nada e
não executa nada do relatório. Relatórios de versões antigas do `ptbr_bench.py` são recusados: rode de novo.

Passar significa que o arquivo é consistente com os dados públicos e com a sua declaração. Não prova que o checkpoint
declarado gerou as respostas, por isso os mantenedores reproduzem o resultado antes de listá-lo.

Envie abrindo uma discussão na página do dataset
(<https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/discussions>) com o relatório anexado, ou use o
[modelo de issue do GitHub](https://github.com/felhen-ai/saracura/issues/new?template=benchmark-submission.yml). Os mantenedores revisam cada envio à
mão; nada é publicado nem aceito automaticamente.

Antes de anexar: o relatório não pode conter texto privado, tokens, nomes de host nem caminhos. Não cole credenciais
nem prompts privados na discussão. Só entram modelos públicos, sem ajuste nas trilhas privadas.

### Trilhas pública e privada, e lm-eval

- As sete tarefas públicas são o que o `ptbr_bench.py` mede e o que a tabela pública lista.
- As trilhas privadas (triagem de e-mail e comentários públicos de uso real, o teste novo de `camara_tema`; definição em
  `research/private_tasks.json`) mantêm textos e rótulos privados. Os mantenedores pontuam por revisão
  manual, nunca por este script.
- `research/lm_eval/ptbr_typed_decisions` é um protocolo separado para o `lm-evaluation-harness` (prompt genérico em
  modelos de linguagem). Os números dele não são comparáveis com os do `ptbr_bench.py` e nunca entram na mesma tabela.

### LLMs de chat pela ponte local

Modelos de chat genéricos não têm servidor System One próprio. O `research/llm_bridge.py` serve um snapshot fixado do
Transformers em `127.0.0.1` com o mesmo contrato `POST /v1/systemone` (núcleo só com a biblioteca padrão; torch e
transformers só são importados ao iniciá-lo). Há dois modos separados: `strict-generation` é o diagnóstico de seguimento
de formato, padrão, com geração gulosa de até 32 tokens e uma única letra maiúscula exata; `constrained-letter` é a
comparação comum planejada para os sete LLMs gerais, mesmo se o smoke strict falhar. Ele faz um forward causal e escolhe
somente entre os IDs das letras legais, com orçamento de um token, logits finitos convertidos para float32 e desempate
pela primeira opção legal. Cada letra precisa ser um token distinto de round-trip exato. Isso é uma escolha discreta,
não probabilidade calibrada; nenhum modo repara, tenta de novo ou usa rótulo ouro.

A receita acima vale aqui: ambiente separado, download explícito e fixado do snapshot (a ponte nunca baixa), início
offline em loopback com `--evidence-output` (arquivo existente nunca é sobrescrito), `--chat-template-kwargs`
obrigatório (a chave que desliga o raciocínio, ou `{}` após inspecionar o template) e `--context` explícito. O prazo
(`--generation-timeout`, 90 s) cobre a requisição inteira e só é checado entre passos de decodificação e entre
perguntas; mantenha o `--timeout` do avaliador maior. Resposta inválida ou prazo vencido devolvem HTTP 200 com `{}`,
registrado como `no_answer` uma única vez. `GET /health` traz o manifesto de execução, inclusive modo, orçamento real,
IDs/tokens brutos e caminho de `logits_to_keep` do modo restrito, além de `evidence_sha256` e dos contadores. Use o
SHA256 dos bytes exatos do arquivo de evidência como `--server-version`; o servidor declarado deve ser exatamente
`ptbr-llm-bridge-constrained-letter` ou `ptbr-llm-bridge-strict-generation`, e o nome também inclui o modo. Capture
`/health` antes e depois da execução completa; contador diferente de zero torna a execução reprovada e sem ranking.

Limites internos dos runtimes devem aparecer na configuração publicada. Laya 0.3.20 corta cada opção em 48 tokens, afetando 599 itens desta rodada. Declare esse cap na evidência e nos metadados de contexto, preserve as descrições originais e confira o contexto integral antes do full. O bloqueio anterior por esse cap fica como histórico; avaliador e ponte não acrescentam cortes além dos 3.500 caracteres comuns.

CPU troubleshooting: use an isolated environment with CPU PyTorch wheels and without FLA/Triton. Reusing a CUDA-optimized environment can make Qwen3.5 select GPU kernels even with `CUDA_VISIBLE_DEVICES` empty; this is a runtime error, not an incorrect model answer. The internal CPU run uses `torch==2.8.0+cpu` from the official PyTorch CPU index and `transformers==5.20.2`. Record actual installed versions in the execution evidence.

Problemas na CPU: use um ambiente isolado com PyTorch para CPU, sem FLA/Triton. Reutilizar um ambiente otimizado para CUDA pode fazer o Qwen3.5 selecionar kernels de GPU mesmo com `CUDA_VISIBLE_DEVICES` vazio; isso é erro de runtime. A rodada interna em CPU usa `torch==2.8.0+cpu` do índice oficial de CPU do PyTorch e `transformers==5.20.2`; registre as versões efetivas na evidência.
