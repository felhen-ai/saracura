---
title: Rodada local comparável de modelos no benchmark PT BR
kind: plan
area: research
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/comparable-round-2026-10-21.md
globalRef: qmd://saracura/docs/action/comparable-round-2026-10-21.md
reviewCadenceDays: 30
lastReviewedAt: 2026-10-21
sourceRefs:
  - https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/tree/00f5d6dcc6679751723289889b8b0b28b71cbcb9
  - https://huggingface.co/docs/hub/en/eval-results
  - https://huggingface.co/docs/hub/leaderboard-data-guide
  - https://github.com/huggingface/huggingface.js/blob/234c367c120753f5e9df3ec30d9c57c687a5e407/packages/tasks/src/eval.ts
related:
  - research/rounds/2026-10-21/manifest.json
  - research/rounds/2026-10-21/historical-audit.json
  - research/evaluation-guide.md
  - research/lm_eval/ptbr_typed_decisions/README.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Rodada local comparável de modelos no benchmark PT BR

Selecionamos 14 checkpoints públicos para comparar sete LLMs de uso geral e sete modelos de decisão nas mesmas sete tarefas e 5.196 itens. Os dados, revisões de pesos e critérios estão fixados no [manifesto](../../research/rounds/2026-10-21/manifest.json). A preparação está concluída; a nova rodada completa ainda não começou. Falta implementar e validar a ponte dos LLMs para as mesmas requisições tipadas e obter disponibilidade de hardware. Não publicar uma classificação nova antes desses passos.

A seleção privilegia modelos de uso difundido, suporte a português, diversidade de fornecedor e tamanhos de até aproximadamente 8B. Downloads e likes do Hub são sinais de adoção, registrados na consulta de 21/10/2026, não provas de qualidade. Todos os 14 têm acesso público sem gate; 13 declaram Apache-2.0 e Phi declara MIT. A execução usa os equipamentos existentes, sem API paga, HF Jobs pago, Runpod ou aluguel de GPU. Há consumo local de energia, disco e tempo.

## Modelos e avaliação existente

| Modelo | Porte | Situação no nosso ranking | Plano |
|---|---:|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 4B | Sem avaliação localizada | Primeira referência geral |
| Qwen/Qwen3-8B | 8B | Sem avaliação localizada | Referência maior; sem thinking |
| Qwen/Qwen3.5-4B | 4B nominal | Sem avaliação localizada | Arquitetura híbrida; somente texto; smoke específico |
| Qwen/Qwen2.5-7B-Instruct | 7B | Sem avaliação localizada | Referência consolidada de outra geração |
| microsoft/Phi-4-mini-instruct | 3,8B | Sem avaliação localizada | Família Microsoft, suporte a português |
| HuggingFaceTB/SmolLM3-3B | 3B | Sem avaliação localizada | Opção compacta com português; sem thinking |
| ibm-granite/granite-3.3-8b-instruct | 8B | Sem avaliação localizada | Família IBM, português; sem thinking |
| jaredpalmer/kev-0.8b | 0,8B | 50,7% histórico | Reavaliar e confirmar base e head |
| jaredpalmer/kev-4b | 4B | 62,2% histórico | Reavaliar com pesos e runtime fixados |
| Mapika/decider-4b | 4B | 64,1% histórico | Reavaliar; o card atual declara v3 de 20/10 |
| interfaze-ai/lev | 4B com adapter | 61,8% histórico | Reavaliar base, LoRA e head como um conjunto |
| convaiinnovations/laya, subfolder multilingual | 322M | 39,2% histórico; prova nova de 41,95% | Âncora reproduzível em CPU; preservar as duas medições identificadas |
| felhen-ai/saracura-ptbr-v0 | 322M | 68,5% histórico | Reavaliar; distinguir modelo ajustado em PT-BR |
| felhen-ai/saracura-ptbr-4b | 4B com adapter | 72,9% histórico HTTP | Reavaliar; distinguir modelo ajustado em PT-BR |

As notas históricas acima vêm do JSON do Space, atualizado em 09/10. Não descrevem necessariamente o checkpoint atual do mesmo repositório. O card do Saracura 4B também mostra 73,1% de outro percurso; a rodada não deve misturar esse valor com 72,9% do servidor HTTP. Modelos ajustados com os splits de treino do benchmark recebem essa identificação na tabela; seus resultados não equivalem a transferência sem ajuste.

O Space contém 16 modelos e três referências. Os 12 relatórios HTTP públicos legados examinados declaram 5.196 itens e zero falhas cada, mas não contêm previsões por item ou checkpoint imutável. São históricos, não resultados validados pelo novo validador. O relatório publicado do Laya em CPU contém 5.196 previsões, checkpoint, protocolo e hashes, com média recalculada de 0,4195 e zero falhas. Essa execução é uma âncora, não motivo para substituir silenciosamente os 39,2% históricos. O [inventário auditado](../../research/rounds/2026-10-21/historical-audit.json) distingue ausência de trace de ausência de execução.

Não escolher os checkpoints depois de ver suas notas nesta rodada. Modelos excluídos inicialmente: APIs fechadas, modelos de dezenas/centenas de bilhões, pesos removidos do Hub e modelos que exigem novo aceite de acesso. Gemma e Llama podem entrar numa segunda edição se o acesso estiver regularizado; não contornar gates com cópias de terceiros. Qwen2.5-3B e Liquid foram considerados, mas não são necessários nesta seleção de 14; não assumir que suas licenças específicas são Apache-2.0.

## Protocolo comparável

Usar `research/ptbr_bench.py` do commit `5c06fd14a3f98cb08158fa02d842b2caabcf242a`, protocolo `ptbr-typed-decisions-v1`, schema 2, dataset no commit `00f5d6dcc6679751723289889b8b0b28b71cbcb9`. O manifesto fixa os sete hashes dos arquivos de teste e as contagens 599, 259, 650, 893, 995, 1.000 e 800. Manter ordem alfabética das tarefas, semente 20261009, embaralhamento por item, estado limitado a 3.500 caracteres e média não ponderada da balanced accuracy por tarefa. A semente de bootstrap é 7, com 1.000 reamostragens.

Os servidores nativos recebem as requisições do avaliador sem alteração. Para LLMs, implementar uma única ponte local que receba exatamente essas requisições, renderize pergunta e opções na ordem recebida e mapeie uma resposta curta de volta à opção original. A ponte não pode reordenar alternativas, ler golds, acrescentar descrições de classes geradas por outro modelo, usar RAG ou mudar o truncamento. Usar template de chat oficial do checkpoint, zero-shot, sem thinking quando suportado. A geração livre original (`strict-generation`, gulosa, até 32 tokens) fica como diagnóstico separado. Revisão r7: os sete LLMs gerais do manifesto serão avaliados com `constrained-letter`, um forward e argmax entre tokens das letras válidas, orçamento de um token. Não selecionar o melhor modo por modelo; resultados dos dois modos ficam em linhas separadas. `checkpoint.server` e o nome do relatório incluem o modo, conferido pelo sidecar contra a evidência imutável. Registrar o template renderizado e as configurações. Respostas inválidas e esgotamento do orçamento contam como falha; não reparar com outro modelo ou refazer por insatisfação com o conteúdo.

Não inventar probabilidades para respostas geradas. Um mapeamento booleano para o contrato `noul` só representa a decisão discreta; não é confiança calibrada. Comparar qualidade das decisões nas tarefas, mostrando método de leitura e template por modelo. Calibração, throughput e velocidade entre métodos são análises separadas. Se for necessário decoding restrito, ele constitui outra configuração identificada; não misturar com geração livre na mesma linha.

A integração `research/lm_eval` existente usa loglikelihood de letras e ordem fixa. Ela pode produzir uma tabela própria, mas suas notas não entram nesta rodada. O script antigo `bench_llm.py` usa outro prompt, até 6.000 caracteres e descrições adicionais; também não serve como execução comparável. A referência Qwen 27B já publicada não conta entre os 14 desta edição.

Antes do primeiro full run, a ponte precisa passar casos sintéticos cobrindo choice de 3 e 24 opções, booleanos, permutação de ordem, caracteres PT-BR, resposta inválida, timeout e ausência de truncamento adicional. Um teste deve comparar as requisições e a sequência de itens dos dois percursos. Revisões do servidor, da ponte, tokenizer, modelo-base, adapter, head e dependências precisam estar fixadas. Os SHAs dos repositórios de modelo no manifesto não substituem esses pins de dependências. Não habilitar execução de código remoto sem inspecionar a revisão fixada.

As referências de classe majoritária e TF-IDF + regressão logística serão recalculadas em CPU e não contam entre os 14 modelos. Usar somente o split de treino na mesma revisão do dataset para ajustar referências; escolhas de hiperparâmetros usam dev, nunca o teste. Fixar os hashes de train/dev, semente, dependências e receita antes da execução. Publicar o orçamento de dados supervisionados junto da nota de TF-IDF, sem apresentá-la como zero-shot. As notas históricas de referências também não são automaticamente revalidadas.

## Hardware e fila de execução

Inspeção read-only de 21/10: Mac M4 Pro com 24 GiB de memória unificada e cerca de 30 GiB livres em disco. Servidor com RTX 5090 de 32 GiB (32.607 MiB reportados), 29.904 MiB ocupados, cerca de 23 GiB de RAM disponível e 28 GiB livres em disco. GPU com 0% de uso instantâneo continua ocupada por um serviço residente. Não descarregar esse serviço nem mudar a política do arbiter para começar o benchmark.

Plano principal: CPU do Mac para os 322M e teste do 0,8B; CUDA do servidor para 3B–8B quando houver slot exclusivo compatível com sua política operacional. Kev documenta MLX em Apple Silicon, mas o smoke precisa confirmar a compatibilidade de cada checkpoint, especialmente o adapter Saracura. MPS/MLX é alternativa condicional; não uma compatibilidade já demonstrada para todos. CPU pode atender um modelo maior se o smoke confirmar memória e duração aceitáveis.

Para os LLMs, o peso bruto em bf16 varia de aproximadamente 5,7 a 15,3 GiB; o manifesto traz estimativas por modelo. São limites inferiores: somar tokenizer, cache KV, ativações, runtime, download parcial e possíveis cópias fp32. Os 4B com adapter ainda precisam do modelo-base e head. Antes de cada download, verificar cache real e espaço para o artefato mais 10 GiB de reserva; se não houver, adiar. Não baixar todos os modelos juntos nem apagar caches de outras tarefas. Uma configuração quantizada, se necessária, terá hash, avaliação e linha próprios; nunca herdará a nota bf16.

Sequência: Laya 322M → Saracura v0 → Kev 0,8B → SmolLM3 → Qwen3 4B → Phi mini → demais LLMs → modelos de decisão 4B. Somente um modelo residente por equipamento durante medições. Executar smoke de dois itens por tarefa (14 chamadas por modelo), medir memória/prefill e estimar duração. Smoke não entra no ranking. Revisão r7: o avaliador e a ponte não podem acrescentar cortes além dos 3.500 caracteres comuns. Pré-processamento intrínseco upstream deve ser declarado: Laya e Saracura-v0 usam cap de 48 tokens por opção, afetando 599 itens; isso é característica da configuração, não corte adicional pelo avaliador. O bloqueio anterior por esse cap fica como registro histórico; avaliações completas são elegíveis com o cap declarado e contexto integral conferido. Se a ponte acrescentar truncamento, tratar como falha de preflight e ajustar a configuração antes do full, sem descartar os itens difíceis.

A rodada completa exige 72.744 chamadas, mais 196 do smoke, sem contar referências. Não estimar prazo a partir das 429 s do Laya: arquiteturas e runtimes diferem. Usar as latências observadas no smoke para orçamento de tempo. Salvar cada relatório com nome exclusivo, validar e conferir 5.196 itens, sete tarefas, zero falhas, hashes, proveniência e CI. Só então publicar; mostrar também a lista de candidatos cujo preflight ou execução falhou, sem substituí-los silenciosamente por modelos com melhor nota.

## Comandos preparados

O arquivo [commands.json](../../research/rounds/2026-10-21/commands.json) contém argumentos completos de smoke e full para os 14 checkpoints. Ele não inicia servidores e não deve ser interpretado como fila automática: os LLMs dependem da ponte acima e cada servidor precisa do próprio preflight. Usa portas locais, revisão de dados fixada, saídas exclusivas e metadados declarados. Substituir os campos `PENDING_PREFLIGHT` com contexto, hardware e versão efetivamente observados antes do full. Depois, executar `python research/validate_submission.py <arquivo-full.json>`. O [guia](../../research/evaluation-guide.md) contém o exemplo de servidor Laya já validado.

## Integração ao ranking nativo do Hugging Face

Estado observado: dataset existe, mas não contém `eval.yaml`; o endpoint de leaderboard responde HTTP 200 com lista vazia. A tag comum `benchmark` não prova registro nativo. A documentação exige `eval.yaml`, um framework reconhecido e, na fase beta, inclusão pelo time HF na allow-list. O identificador proposto `ptbr-typed-decisions` não aparece no enum de frameworks consultado. [Registro e formato](https://huggingface.co/docs/hub/en/eval-results); [enum fixado](https://github.com/huggingface/huggingface.js/blob/234c367c120753f5e9df3ec30d9c57c687a5e407/packages/tasks/src/eval.ts).

Preparar o registro do nosso avaliador, preservando o protocolo. Não declarar `inspect-ai`, `lm-eval` ou outro framework para contornar a validação. O [eval.yaml proposto](../../research/rounds/2026-10-21/eval.yaml.proposed) depende do registro do framework e de confirmação de como representar as sete tarefas, a média agregada e a escala da métrica; não foi enviado ao Hub. Usar IDs diferentes se a ponte exigir um protocolo semanticamente diferente.

Resultados nativos ficam em `.eval_results/*.yaml` dos repositórios de modelo, com dataset/task, valor, revisão, data e evidência. Nos modelos externos, preparar PRs individuais depois de medir e obter autorização de envio. Nosso validador não emite `verifyToken`; não atribuir selo HF `verified` a execuções locais. O contato com HF fica em [rascunho](../../research/rounds/2026-10-21/hf-registration-request.txt), sem envio. [Formato dos resultados](https://huggingface.co/docs/hub/en/eval-results).

A validação final da integração exige o dataset reconhecido como benchmark, uma avaliação retornada pelo endpoint `/leaderboard` e o embed renderizando esse resultado. Até essa confirmação, o Space existente continua sendo a publicação operacional. Não inferir integração concluída por um commit de YAML.

## Próximas entregas

1. Implementar e testar a ponte local comum; fixar runtimes e dependências após inspeção das interfaces upstream.
2. Executar os 196 itens de smoke, registrar memória, contexto e tempo, e fechar a alocação de cada modelo.
3. Executar os 14 conjuntos completos sequencialmente; recalcular métricas e publicar relatórios reproduzíveis antes de convidar autores a reproduzir.
4. Registrar o framework e solicitar habilitação nativa ao HF com o protocolo e as evidências, após autorização do contato externo.
5. Usar a classificação completa para selecionar os modelos do estudo de transferência a casos reais, preservando os testes reservados desse estudo.
