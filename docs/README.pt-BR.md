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
reviewCadenceDays: 60
lastReviewedAt: 2026-10-05
sourceRefs:
  - https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench
  - https://huggingface.co/felhen-ai/saracura-ptbr-v0
related:
  - README.md
  - docs/decisions/0005-open-base-benchmark-first.md
  - research/README.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura

[English](../README.md) | [Português (Brasil)](README.pt-BR.md)

Decisões tipadas sobre texto em português do Brasil: um benchmark aberto e modelos de decisão pequenos e rápidos,
ajustados para PT-BR.

> Português do Brasil de verdade. Dados abertos. Medido no mesmo benchmark que todo o resto.

## O que existe hoje

| Artefato | Onde | O que é |
|---|---|---|
| **Benchmark** | [`felhen-ai/ptbr-typed-decisions-bench`](https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench) | 7 tarefas de decisão tipada sobre texto nativo em PT-BR, de fontes com licença permissiva e rótulos humanos (proposições legislativas, jurisprudência do TCU, resumos científicos, comentários de redes sociais, alegações checadas, respostas de FAQ). Splits de treino, validação e teste. |
| **Saracura PT-BR v0.1** | [`felhen-ai/saracura-ptbr-v0`](https://huggingface.co/felhen-ai/saracura-ptbr-v0) | Modelo de decisão de 322M de parâmetros, ajustado a partir do [Laya](https://github.com/NandhaKishorM/laya) multilíngue. Decisões de escolha e sim/não em uma passada, cerca de 3 ms por decisão em GPU, usável em CPU. |

Um checkpoint maior (4B de parâmetros, com a receita do [Kev](https://github.com/jaredpalmer/kev)) está em treino e
será publicado com o mesmo nome.

## Como usar o modelo

```bash
pip install laya
```

```python
import laya

agent = laya.load("felhen-ai/saracura-ptbr-v0")

resultado = agent.predict(
    "Requer informações ao Ministro da Saúde sobre a distribuição de vacinas nos municípios do interior.",
    {
        "tema": {
            "type": "choice",
            "instructions": "Qual é o tema desta proposição legislativa?",
            "criteria": {"saude": "Saúde", "educacao": "Educação", "economia": "Economia"},
        }
    },
)
print(resultado["answers"]["tema"]["choice"])  # saude
```

Sem conta, sem servidor, sem download além dos pesos. O model card traz os resultados completos, os dados de treino e
as limitações.

## Resultados (acurácia balanceada, split de teste do benchmark)

| Modelo | Média das 7 tarefas |
|---|---:|
| Classe mais comum | 27,6% |
| `convaiinnovations/laya-multilingual` (sem ajuste) | 39,2% |
| `telepatia-ai/laya-pt-es-typed` (ajustado em dados traduzidos) | 44,6% |
| TF-IDF + regressão logística, um classificador por tarefa | 64,0% |
| `Qwen/Qwen3.8-27B` sem ajuste, com descrições das classes (versão anterior do benchmark) | 66,2% |
| **Saracura PT-BR v0.1 (322M, um modelo só)** | **68,5%** |

Em perguntas e textos que o modelo nunca viu (gerados por um professor de pesos abertos), a v0.1 concorda com o
professor em 86,8% dos casos em textos reais e em 89,6% em textos sintéticos de oito casos de uso (triagem de e-mail,
roteamento de tickets, qualificação de leads, intenção em WhatsApp, documentos financeiros, avaliações de produto,
moderação, supervisão de agente). São números de concordância, não de acerto contra rótulo humano.

## Avalie o seu modelo

As linhas do benchmark são JSONL no formato de requisição do Laya: `text`, `type` (`choice` ou `noul`),
`instructions`, `criteria` e `gold`. O `research/bench.py` avalia qualquer checkpoint compatível com o Laya e os
baselines clássicos:

```bash
uv sync --dev
uv run python research/bench.py baselines
uv run python research/bench.py eval --name meu --model sua-org/seu-checkpoint --device cpu
```

Reporte acurácia balanceada com a ordem das opções embaralhada por item, como os scripts fazem, para os resultados
continuarem comparáveis.

## Como os modelos são treinados

Está tudo em [`research/`](../research/README.md): a montagem do benchmark a partir das fontes, a geração de
perguntas variadas com professores de pesos abertos, o ajuste com as receitas do Laya e do Kev e os scripts de
avaliação. Dados de treino que não são redistribuídos (anúncios de marketplace, documentos internos) estão declarados
nos model cards.

A direção e seus motivos estão na [ADR 0005](decisions/0005-open-base-benchmark-first.md). As ADRs anteriores e os
pacotes `src/` e `benchmarks/` pertencem ao runtime de pesquisa que a precedeu; ficam como API tipada e registro
histórico.

## Como contribuir

A contribuição mais valiosa é uma tarefa com texto nativo em PT-BR, rótulos humanos e licença que permita
redistribuição. Abra uma issue com a fonte, a licença e como os rótulos foram produzidos. Contribuições seguem o
[CONTRIBUTING.md](../CONTRIBUTING.md); leia o [SECURITY.md](../SECURITY.md) antes de adicionar carregamento de modelo
ou dataset.

## Licença

Apache-2.0. Veja [LICENSE](../LICENSE). A compilação do benchmark é CC BY 4.0, com cada tarefa sob a licença da
fonte, listada no card do dataset.
