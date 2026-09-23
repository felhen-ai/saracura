---
title: Saracura — leia-me em português brasileiro
kind: reference
area: product
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/README.pt-BR.md
globalRef: qmd://saracura/docs/README.pt-BR.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-22
sourceRefs: []
related:
  - README.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura

[English](../README.md) | [Português (Brasil)](README.pt-BR.md)

Saracura é um motor de pesquisa PT-BR-first e local para decisões tipadas em volume. A tese arquitetural é simples: codificar um estado uma vez e responder muitas perguntas calibradas com baixo custo incremental.

> Um estado. Muitas decisões calibradas. Aberto e local.

Este repositório ainda é um **alpha exclusivo para pesquisa**. Ele não contém pesos treinados, não afirma qualidade de decisão e não deve ser usado como gate de automação ou autorização. O backend determinístico incluído é apenas uma fixture para exercitar o contrato e os invariantes do runtime.

PT-BR-first significa que o português brasileiro é o primeiro idioma dos exemplos e do trabalho de governança de dados, anotação e avaliação. Isso não significa que este alpha já distribua um checkpoint otimizado para PT-BR. A API e a arquitetura permanecem neutras em relação ao idioma para que o mesmo protocolo de evidência possa se expandir para inglês e outros idiomas.

O inglês é o idioma canônico da documentação técnica. O quickstart, os exemplos públicos e os materiais de avaliação em PT-BR também são mantidos em português brasileiro quando aplicável.

## Escopo atual

- contratos fechados de request, response e erro `v1alpha1`;
- perguntas `choice` em workflows conhecidos e imutáveis;
- normalização NFC versionada antes da canonicalização RFC 8785;
- segmentos semânticos prefixados por comprimento;
- uma única codificação do estado reutilizada em Q=1, Q=10 e Q=50;
- testes de isolamento entre perguntas com tolerância numérica absoluta de `1e-12`;
- chaves de cache de schema fail-closed com os bytes canônicos completos e eixos de revisão;
- artefatos de calibração imutáveis e compatíveis de forma fail-closed;
- runtime in-process e CLI locais;
- um backend experimental opt-in `laya-universal` para o workflow Choice dinâmico exato;
- trilhas opt-in de pesquisa para aquisição de encoder, treino de head sintético, calibração humana em PT-BR e controles externos.

Não estão incluídos no runtime instalado: downloads de modelos incluídos no pacote ou disparados por request, datasets ou checkpoints incluídos, labels dinâmicos fora do workflow Choice experimental exato da Fase 4D, heads boolean ou ordinal, servidor HTTP, fallback remoto, telemetria ou automação de produção. As ferramentas de pesquisa podem adquirir snapshots revisados de encoders e treinar heads experimentais locais apenas por meio de fluxos explícitos, controlados pelo operador e offline-first.

## Arquitetura de pesquisa em duas camadas

Saracura está seguindo uma única API de decisões tipadas com duas camadas de execução. A camada experimental `universal` deve aceitar novos schemas Choice sem exigir que usuários adotem pacotes de modelos predefinidos ou heads específicos por tarefa. A camada opcional `compiled` especializa decisões estáveis e de alto volume quando essa otimização se justifica. A primeira implementação universal continua experimental, nenhum modelo está aprovado para produção e resultados não autorizam automação. TypeSafe/Jev é referência de benchmark e de design, não uma dependência de runtime ou fallback do Saracura. Veja o [ADR 0001](decisions/0001-two-tier-decision-architecture.md).

## Instalação e validação

É necessário usar Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

O ambiente de desenvolvimento padrão não instala backend de modelo nem dependências pesadas de ML.

## Executar a fixture PT-BR autoescrita

```bash
uv run saracura decide \
  --request examples/ptbr-support-request.json \
  --calibration examples/ptbr-support-calibration.json
```

O resultado distingue scores crus de probabilidades normalizadas da fixture e declara `fixture_only`. O artefato exercita as verificações de compatibilidade, mas não foi ajustado em dados de calibração; ele não prova que o backend seja preciso, calibrado ou seguro para automação. `verified_for_research` fica reservado a um artefato futuro que cumpra seu protocolo held-out declarado.

## Boundary

Saracura é independente. Instalação, testes, CLI e exemplos não exigem código, serviço, dado ou configuração privados. Todo o conteúdo de fixture desta fase é autoescrito e declarado em `benchmarks/manifests/ptbr-fixture-v1.json`.

Consulte [SECURITY.md](../SECURITY.md) antes de adicionar carregamento de modelos, tokenizers, datasets ou plugins. Contribuições devem seguir [CONTRIBUTING.md](../CONTRIBUTING.md).

As licenças e versões resolvidas das dependências de runtime estão registradas em [dependency-licenses.md](dependency-licenses.md).

## Executar o harness local de escala

O runner de benchmark, usado apenas a partir do checkout, consome a fixture autoescrita e executa o `DecisionEngine` público em Q=1, Q=10 e Q=50. Ele grava amostras brutas e um relatório Markdown em `.artifacts/`, que é ignorado pelo Git:

```bash
uv run python -m benchmarks.run \
  --suite decision-scaling \
  --backend fixture \
  --warmups 1 \
  --iterations 10 \
  --code-revision local-smoke \
  --output-dir .artifacts/benchmarks
```

O schema do resultado é `phase2a.v1`. Os percentis usam nearest-rank (`sorted[ceil(p*n)-1]`) e cada amostra registra o digest das respostas e a observação de uma única codificação do estado. Esta fixture é apenas um instrumento de pesquisa: seu tempo não é performance de modelo treinado, não prova qualidade e não autoriza automação ou decisões de produção. O runner não faz parte do wheel instalado e não baixa modelos, tokenizers, datasets nem runtimes pesados de ML.

## Gate opcional de pesquisa de encoders

A Fase 2B é uma trilha local e explícita de pesquisa. Ela não escolhe modelo,
treina, calibra, mede qualidade nem autoriza automação. O ambiente padrão
continua leve e offline. Depois da validação padrão, o operador pode instalar o
extra isolado e adquirir um candidato revisado em revisão imutável:

```bash
uv sync --locked --dev --extra encoder-eval
uv run python -m benchmarks.encoder_gate validate-registry
uv run python -m benchmarks.encoder_gate acquire \
  --candidate multilingual-minilm-l12 --allow-network
uv run python -m benchmarks.encoder_gate probe \
  --candidate multilingual-minilm-l12 --device cpu --output-dir .artifacts/encoders
```

A aquisição nunca é disparada por uma requisição. Ela aceita somente um ID do
registry, verifica cada byte contra o manifesto e não substitui um snapshot
imutável. O `mmbert-base` fica bloqueado porque a revisão analisada publica
`pytorch_model.bin` sem peso safetensors. Os relatórios são observações somente
do encoder: não medem qualidade da decisão, head treinado, calibração, latência
ponta a ponta ou prontidão para automação.

## Roteamento MiniLM local opt-in da Fase 4A

A Fase 4A pode executar um único head congelado, sintético e de cinco labels
para roteamento de suporte quando o operador fornece explicitamente um snapshot
local já verificado, o manifesto de treino selado e o checkpoint safetensors.
Não há download, descoberta por cache, variável de ambiente, requisição ou
serviço de rede. A instalação padrão não muda; o opt-in é explícito:

    uv sync --locked --dev --extra local-minilm
    uv run saracura describe-backend --backend minilm-routing \
      --encoder-snapshot <snapshot-verificado> \
      --training-manifest <training-manifest.json> \
      --checkpoint <checkpoint.safetensors> --device cpu

O loader verifica cada byte fornecido por descritores locais, constrói BERT e o
tokenizer diretamente desses bytes e vincula a ABI de runtime/dispositivo à
revisão imutável do modelo. O artefato de identidade da Fase 4A continua
fixture_only: não tem ajuste, avaliação, limiar ou autorização de automação.
MPS da Apple é uma escolha explícita do operador e não tem fallback automático
para CPU.

## Choice universal experimental da Fase 4D

`laya-universal` é um backend local, opt-in e somente para pesquisa para o
único workflow Choice dinâmico `universal-choice@phase4d-laya.v1`. Ele exige o
grupo opcional isolado de dependências, um snapshot Laya local e imutável
fornecido pelo operador e um dispositivo CPU ou MPS explícito; nunca baixa,
descobre nem contata um provedor de modelo. A requisição sintética em PT-BR em
[`examples/ptbr-universal-request.json`](../examples/ptbr-universal-request.json)
usa a triagem de e-mail somente como o próximo piloto em modo sombra e não
contém dados de caixa postal.

```bash
uv sync --locked --dev --extra universal-local
uv run saracura describe-backend --backend laya-universal \
  --model-snapshot <snapshot-local-laya-verificado> --device cpu
uv run saracura decide --backend laya-universal \
  --request examples/ptbr-universal-request.json \
  --model-snapshot <snapshot-local-laya-verificado> --device cpu
```

A resposta é explicitamente `uncalibrated`, `abstained` e
`automation_allowed=false`. Seus valores normalizados são pesos de ranking, não
confiança ou permissão para arquivar, apagar, mover, responder, encaminhar ou
fazer qualquer outra alteração em e-mail. O candidato permanece
`research_only_unresolved_provenance`; um smoke local bem-sucedido demonstra
somente compatibilidade de execução, não qualidade, calibração, licenciamento
ou prontidão para produção.

## Gate de dados, licença e privacidade da Fase 2C

Antes de criar qualquer pacote de treino, o repositório valida o registry de
políticas de origem, que contém apenas metadata:

```bash
uv run python -m benchmarks.data_policy_gate validate-registry
```

O registry tem exatamente oito categorias e não aprova nenhum byte de dataset.
Somente casos futuros, originalmente escritos por humanos, e derivados
determinísticos poderão ser considerados para treino PT-BR, sempre sob um
manifesto de artefato separado e revisão humana de privacidade e direitos. O
Amazon MASSIVE oficial em `pt-PT` fica restrito a um controle externo separado;
ele não pode virar evidência PT-BR. Fontes assistidas por modelo, privadas, de
clientes e públicas sem procedência permanecem em quarentena ou bloqueadas.
Este é um gate de engenharia, não aconselhamento jurídico nem prova de
qualidade do dataset. A validação não baixa dados nem importa um loader de
datasets.

## Gate de pacote first-party da Fase 3A

O protocolo de anotação de roteamento de suporte e o gate de divisão determinística são apenas infraestrutura offline. Nenhum registro de dataset é incluído, e os comandos não autorizam treinamento, alegações de qualidade, publicação ou automação:

```bash
uv run python -m benchmarks.first_party_gate validate-protocol
uv run python -m benchmarks.validate_manifests
```

Mantenedores futuros deverão fornecer um JSONL canônico de estados-base, atestado externamente, e uma seed pública para construir um plano de divisão que não sobrescreve artefatos existentes. Autoria humana, privacidade, direitos, representatividade e revisão independente permanecem fora do alcance da ferramenta.

## Benchmark de throughput ponta a ponta da Fase 3C

A Fase 3C é um instrumento de performance usado somente a partir do checkout e apenas com dados sintéticos. Ela mede o head MiniLM revisado desde o texto em PT-BR, passando por tokenização, inferência Apple MPS, pooling e transferência para CPU, até a materialização das escolhas, com evidências separadas de carregamento e warm path. Requisições opcionais de controle com Jev usam o modelo fixado `typesafe/jev-1.13` pelo OpenRouter somente quando habilitadas explicitamente.

O ambiente padrão permanece offline e leve: importar o runner não importa Torch nem Transformers. A execução real exige um snapshot local verificado do encoder, manifestos selados da Fase 3B e um orçamento aprovado explicitamente; ela não deve autorizar automação nem fundamentar alegações gerais de qualidade. Os testes offline do protocolo podem ser executados com:

```bash
uv run pytest -q tests/test_e2e_benchmark.py
uv run ruff check benchmarks/e2e_benchmark.py tests/test_e2e_benchmark.py
uv run mypy benchmarks/e2e_benchmark.py tests/test_e2e_benchmark.py
```

## Benchmark de controle nativo TypeSafe da Fase 3D

A Fase 3D é um experimento explícito somente com dados sintéticos. Ela reutiliza o workload selado e o construtor da Fase 3C, fixa `jev-1.13.0` e mantém separados probabilidade, confiança, tokens, latência e custo calculado nativos. O ambiente padrão continua offline e leve, sem SDK TypeSafe ou backend de runtime. Uma execução ao vivo exige o contrato genérico `TYPESAFE_API_KEY`, `--allow-network` e o orçamento local revisado `0.25`. O custo calculado não é comprovante de cobrança e nenhum resultado escolhe limiar de automação.

```bash
uv run pytest -q tests/test_typesafe_native.py
uv run ruff check benchmarks/typesafe_native.py tests/test_typesafe_native.py
uv run ruff format --check benchmarks/typesafe_native.py tests/test_typesafe_native.py
uv run mypy benchmarks/typesafe_native.py tests/test_typesafe_native.py
```

## Licença

Apache License 2.0. Consulte [LICENSE](../LICENSE).
