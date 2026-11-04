---
title: Official-topic and full-input benchmark V3
kind: reference
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: research/v3_spec.md
globalRef: qmd://saracura/research/v3_spec.md
reviewCadenceDays: 60
lastReviewedAt: 2026-11-03
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

# Benchmark V3

A V3 usa os temas oficiais da Câmara e o texto integral do campo de origem na tarefa jurídica. Os dados, resultados e protocolos V1/V2 permanecem disponíveis como histórico, sem sobrescrita.

## Câmara: respostas oficiais

A tarefa passa a pedir um dos temas oficiais da proposição e aceita qualquer tema atribuído a ela pela Câmara. As alternativas são os 32 temas do catálogo oficial congelado em 3 de novembro de 2026. O texto é a ementa do CSV oficial, sem corte de caracteres; não é o inteiro teor da proposição.

O conjunto mantém 529 dos 599 textos anteriores, vinculados por correspondência exata de texto normalizado a uma única proposição com temas oficiais. Outros 70 exemplos vêm do mesmo snapshot. A normalização aplica NFC, colapsa espaços Unicode e remove espaços nas extremidades. Textos e IDs conhecidos de treino e desenvolvimento são excluídos. Não há correspondência aproximada nem rotulagem manual. A ausência de IDs em parte da origem histórica impede afirmar independência semântica completa.

As reposições excluem todos os textos do teste anterior e textos atribuíveis a múltiplos IDs. A seleção tenta alcançar cinco exemplos por tema disponível, escolhendo primeiro o candidato que cobre mais temas ainda abaixo desse mínimo; desempata por SHA-256 de `42:{proposalID}` e ID numérico. Completa as 599 linhas pela mesma ordem de hash. A ordem das alternativas usa uma sequência própria com semente 42.

O snapshot elegível cobre 31 temas. Ciências Exatas e da Terra permanece entre as 32 alternativas, mas não tem exemplo elegível: os três registros disponíveis são ambíguos ou coincidem com treino/desenvolvimento. A documentação e os resultados informam essa ausência e os suportes de cada tema.

## Pontuação

A tarefa mede acerto de um tema válido, com balanceamento pelos temas representados. Não mede recuperação de todos os temas de uma proposição. Para cada tema com exemplos, calcula a proporção de respostas que pertencem ao conjunto oficial aceito; depois tira a média desses temas. Um mesmo acerto pode contribuir para os vários temas oficiais daquele exemplo.

Se `M` é o número de temas com exemplos, `K = 32` o número de alternativas, `S_i` o conjunto aceito e `n_c` o suporte do tema, o peso de cada exemplo é `w_i = sum(1/n_c for c in S_i) / M`. A soma dos pesos é um. A nota bruta balanceada é `sum(w_i * acerto_i)`. O acaso uniforme é `B = sum(w_i * len(S_i) / K)`. A nota normalizada é `100 * max(0, min(1, (nota_bruta - B) / (1 - B)))`.

Cada relatório também mostra acerto simples, suportes, temas sem exemplos e o melhor preditor constante. As seis outras tarefas conservam a pontuação da V2. O índice continua sendo a média das sete tarefas. Os dois Saracuras permanecem separados da classificação externa.

As 32 alternativas exigem uma nova avaliação da Câmara em todos os 14 modelos. As previsões antigas com 24 alternativas não substituem essa rodada. O adaptador de LLM V3 usa os identificadores sensíveis a caixa `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef`; o protocolo anterior fica preservado nas demais tarefas.

## Diagnóstico do corte de texto jurídico

Todos os 429 textos que excedem 3.500 caracteres são avaliados em duas condições: prefixo de 3.500 caracteres e texto completo do campo de origem. Kev-4B e Qwen3.5-4B usam as mesmas revisões, precisão, runtime e ordem de alternativas dentro de cada comparação. A ordem original é reproduzida avançando a sequência aleatória pelas sete tarefas originais, em ordem alfabética, com semente 20261009.

A mudança na acurácia balanceada da tarefa inteira é calculada pelos acertos que mudam nesses 429 casos, usando os suportes originais dos 893 exemplos. Não se apresenta essa conta como uma nova execução completa dos 893. Também se informa a concordância entre o braço de prefixo e as previsões da V2, para expor possíveis diferenças de runtime.

O diagnóstico pré-definiu um limiar operacional de três pontos percentuais na tarefa jurídica para decidir se uma nova rodada completa seria necessária. Não era um padrão científico nem um teste de ausência de efeito. As mudanças observadas foram +1,6429 ponto no Qwen3.5-4B e +0,4820 ponto no Kev-4B. No Qwen, 27 respostas passaram de erradas para certas e nove fizeram o caminho contrário; no Kev, foram 20 e 15. Cada comparação usou os 429 pares completos. A concordância do braço de prefixo com a execução V2 foi 98,8345% nos dois modelos.

Após esses resultados, a decisão dos mantenedores em 3 de novembro de 2026 foi adotar texto completo na V3, independentemente daquele limiar. O conteúdo cortado ajudou em parte dos casos e a nova versão elimina esse corte arbitrário. Essa decisão exige uma execução completa dos 893 exemplos jurídicos por modelo, com o mesmo runtime dentro da execução. O diagnóstico não substitui essa rodada: não se misturam as 464 previsões antigas com as 429 novas para produzir um resultado publicado.

Um modelo que não consiga processar os textos completos fica fora da classificação V3, com a limitação de contexto ou de execução documentada; seus resultados V2 permanecem disponíveis. Não se corta, resume ou divide um texto para declará-lo completo. A conclusão do diagnóstico se limita aos modelos, casos e configurações medidos, sem afirmar ausência geral de efeito.

## Execução e publicação

A execução usa um pod dedicado com teto de custo, coleta periódica das evidências e confirmação de exclusão. Um controle local de orçamento e um watchdog destacado no próprio pod limitam o tempo de uso. O watchdog usa apenas as credenciais restritas ao pod fornecidas pelo Runpod e precisa demonstrar a autoparada antes da avaliação. Não se depende de flags de expiração que o provedor não aplica.

Fontes, ordens de alternativas, revisões, configurações, hashes e tentativas parciais são preservados. A publicação da V3 depende de relatórios completos e verificáveis. Os resultados V1/V2 não são sobrescritos. A resposta ao comentário público será preparada com as conclusões verificadas e submetida à aprovação dos mantenedores antes do envio.

## Limites da configuração avaliada

A checagem de tokenização dos 893 casos jurídicos encontrou um caso com 11.477 tokens na sequência nativa completa de Laya multilingual e Saracura v0. Seus encoders ModernBERT declaram 8.192 posições, sem configuração de escalonamento RoPE. Nessa configuração avaliada, ambos ficam sem índice completo V3; seus resultados históricos e os 599 casos novos da Câmara são preservados. Esse registro não afirma que toda futura configuração de extrapolação de contexto seja impossível.
