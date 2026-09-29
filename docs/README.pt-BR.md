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
lastReviewedAt: 2026-09-29
sourceRefs: []
related:
  - README.md
  - docs/decisions/0004-model-first-product-direction.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura

[English](../README.md) | [Português (Brasil)](README.pt-BR.md)

Saracura é um projeto aberto e local-first de modelo de decisão para decisões tipadas rápidas em escala, com avaliação PT-BR-first e API neutra em relação ao idioma.

> Modelos locais pequenos. Muitas decisões tipadas. Evidência em hardware real.

O Saracura está atualmente em **preview de desenvolvimento orientado ao modelo**. O runtime público e as ferramentas de avaliação já existem; o primeiro checkpoint próprio e de uso geral do Saracura é o próximo marco de release e ainda não foi publicado. Até que esse checkpoint e suas medições estejam disponíveis, o repositório não faz alegação comparativa de qualidade e não deve ser usado como gate de automação ou autorização.

## O que estamos construindo

O Saracura está sendo construído como um modelo que qualquer pessoa possa baixar e executar, não como wrapper de outro modelo de decisão nem como laboratório de benchmark cujo principal resultado seja infraestrutura.

O projeto se concentra na interseção de quatro propriedades:

- modelos pequenos que rodem localmente em CPU, Apple Silicon ou GPU modesta;
- decisões tipadas sobre novos schemas por meio de uma API estável;
- workloads de alto volume, incluindo muitas decisões sobre contexto compartilhado;
- treino e avaliação PT-BR-first sem tornar a arquitetura específica de um idioma.

O próximo marco público é um checkpoint próprio do Saracura com:

- quickstart local em um comando;
- pesos públicos e imutáveis, acompanhados de model card;
- medições reproduzíveis de qualidade em PT-BR e de desempenho em hardware;
- uma alegação delimitada pelo resultado que o modelo efetivamente demonstrar.

Calibração, predição seletiva e segurança para automação continuam sendo marcos importantes posteriores. Eles não bloqueiam a publicação de um preview de modelo útil e honestamente delimitado.

## Release atual

O pacote atual fornece a API de decisões tipadas, CLI local, contratos estritos de request e response, ferramentas de benchmark e a fundação de runtime para o próximo modelo. Ele ainda **não** distribui o checkpoint público do Saracura descrito acima.

Releases anteriores integraram checkpoints de terceiros e controles externos para validar o runtime e entender onde as abordagens existentes falham. Essas integrações continuam como evidência reproduzível de pesquisa, mas não são o produto Saracura, não são recomendadas como seu modelo padrão e não definem o roadmap público.

O boundary permanente do produto está registrado no [ADR 0004: direção de produto model-first](decisions/0004-model-first-product-direction.md).

## Roadmap

As GitHub Issues são o roadmap operacional público. O [milestone v0.2.0 Model Preview](https://github.com/felhen-ai/saracura/milestone/1) mostra quanto falta para o primeiro release de modelo; o [issue de roadmap #49](https://github.com/felhen-ai/saracura/issues/49) registra a ordem das dependências e a definição de pronto.

Cada issue do roadmap deve terminar em um artefato visível de modelo, avaliação, empacotamento ou release. O trabalho de automação calibrada é acompanhado separadamente e não dilui o progresso rumo à v0.2.0.

## Arquitetura

O Saracura expõe uma única API de decisões tipadas com dois papéis de modelo:

- `universal`: o modelo Saracura padrão para novos schemas Choice;
- `compiled`: uma especialização opcional para workflows estáveis, repetidos e de alto volume quando as medições a justificarem.

O primeiro release de modelo tem como alvo o papel universal. Uma rota compiled só tem valor quando melhora a fronteira entre qualidade e vazão para um workload declarado; ela não é motivo para atrasar o checkpoint universal.

Consulte o [ADR 0001](decisions/0001-two-tier-decision-architecture.md) para a arquitetura em duas camadas e o [ADR 0002](decisions/0002-open-model-runtime-and-readiness.md) para os limites de evidência e release.

## Instalação para desenvolvimento

É necessário usar Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/felhen-ai/saracura.git
cd saracura
uv sync --dev
uv run pytest -q
```

O ambiente padrão continua leve e não baixa pesos nem dependências pesadas de ML. A aquisição de modelos é sempre explícita e pinada por revisão.

## Exercitar o contrato atual da API

O repositório inclui uma fixture PT-BR autoescrita para que contribuidores exercitem o contrato tipado antes de baixar artefatos de modelo:

```bash
uv run saracura decide \
  --request examples/ptbr-support-request.json \
  --calibration examples/ptbr-support-calibration.json
```

Essa fixture é infraestrutura de teste, não um modelo treinado nem evidência de qualidade de decisão. Este README substituirá a seção pelo quickstart do modelo quando o checkpoint Saracura for publicado.

## Evidência, não marketing de concorrentes

O Saracura mantém registros reproduzíveis de pesquisa, inclusive resultados negativos. Modelos de terceiros só devem aparecer nesses registros como controles sob o mesmo protocolo declarado. A página principal não posiciona o Saracura como adapter ou pequena variação de um checkpoint externo específico.

Registros relevantes:

- [protocolo de prontidão e avaliação de modelos abertos](action/specs/phase5-open-model-benchmark-and-public-readiness.md);
- [protocolo do benchmark nativo em PT-BR](action/specs/phase5d-native-ptbr-benchmark.md);
- [resultado histórico da comparação cega](action/phase4e-comparison-result.md);
- [resultado da avaliação shadow somente leitura](action/phase4f1-shadow-evaluation-result.md).

Resultados históricos de benchmark não estabelecem que o próximo modelo Saracura seja melhor. Uma alegação comparativa só se torna pública quando existirem um checkpoint próprio do Saracura e evidência diretamente comparável.

## Guardrail de produto

Novos adapters, harnesses, integrações de candidatos ou infraestrutura de benchmark devem contribuir diretamente para pelo menos um destes resultados:

1. selecionar ou treinar o checkpoint Saracura;
2. medir qualidade, footprint, latência ou vazão do modelo;
3. empacotar e executar o modelo localmente;
4. melhorar uma fraqueza medida no modelo publicado.

Trabalho que não satisfaça nenhum desses resultados não está no caminho crítico do release do modelo. Isso evita que a infraestrutura auxiliar vire o produto por acidente.

## Segurança e independência

- O Saracura continua utilizável sem Felhen, AIOS, serviços privados, dados privados ou configuração privada.
- Nenhuma requisição dispara download implícito de modelo ou fallback remoto.
- Scores não calibrados não são apresentados como confiança.
- Um preview de modelo não autoriza ações automáticas de alto risco.
- Registros de treino, calibração e avaliação held-out permanecem separados nos protocolos declarados.

Consulte [SECURITY.md](../SECURITY.md) antes de acrescentar carregamento de modelo, tokenizer, dataset ou plugin. Contribuições seguem [CONTRIBUTING.md](../CONTRIBUTING.md).

## Licença

Apache-2.0. Consulte [LICENSE](../LICENSE).
