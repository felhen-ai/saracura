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
lastReviewedAt: 2026-09-21
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

Saracura é um motor experimental e local para decisões tipadas. A tese arquitetural é simples: codificar um estado uma vez e responder muitas perguntas calibradas com baixo custo incremental.

> Um estado. Muitas decisões calibradas. Aberto e local.

Este repositório ainda é um **scaffold alpha exclusivo para pesquisa**. Ele não contém pesos treinados, não afirma qualidade de decisão e não deve ser usado como gate de automação ou autorização. O backend determinístico incluído é apenas uma fixture para exercitar o contrato e os invariantes do runtime.

Português brasileiro é o primeiro idioma das fixtures, enquanto a API e a arquitetura permanecem neutras em relação ao idioma.

## Escopo atual

- contratos fechados de request, response e erro `v1alpha1`;
- perguntas `choice` em workflows conhecidos e imutáveis;
- normalização NFC versionada antes da canonicalização RFC 8785;
- segmentos semânticos prefixados por comprimento;
- uma única codificação do estado reutilizada em Q=1, Q=10 e Q=50;
- testes de isolamento entre perguntas com tolerância numérica absoluta de `1e-12`;
- chaves de cache de schema fail-closed com os bytes canônicos completos e eixos de revisão;
- artefatos de calibração imutáveis e compatíveis de forma fail-closed;
- runtime in-process e CLI locais.

Não estão incluídos: download de modelos, treino, datasets externos, labels dinâmicos, heads boolean ou ordinal, servidor HTTP, fallback remoto, telemetria ou automação de produção.

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
