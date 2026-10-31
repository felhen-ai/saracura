---
title: Auditoria de treino e preparação de dois pilotos de transferência
kind: report
area: research
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/transfer-audit-and-pilots-2026-10-31.md
globalRef: qmd://saracura/docs/action/transfer-audit-and-pilots-2026-10-31.md
reviewCadenceDays: 30
lastReviewedAt: 2026-10-31
sourceRefs:
  - research/combo.py
  - research/kev_data.py
  - research/private_tasks.json
  - research/rounds/2026-10-31-transfer-pilots/training-audit.json
  - research/rounds/2026-10-31-transfer-pilots/source-discovery.json
related:
  - docs/action/comparable-round-2026-10-21.md
  - research/rounds/2026-10-31-transfer-pilots/pilot-protocol.json
supersedes: []
supersededBy: []
sensitivity: public
---

# Auditoria e pilotos: estado verificável

Auditoria documental concluída para os dois checkpoints públicos exatos da rodada dos 14. Foram examinados os model cards nas revisões fixadas, as receitas de preparação, as definições das provas privadas e a disponibilidade dos insumos no checkout local. Não houve acesso à VM 151, carregamento de modelos, contratação de GPU, treino ou inferência. O resultado é um mapa de exposição e dois protocolos preparados; nenhum item foi certificado como independente e nenhuma nova pontuação foi produzida.

## O que o treino já inclui

| Fonte | Evidência | Consequência |
|---|---|---|
| Sete tarefas públicas, split de treino | Ambas as receitas e ambos os cards | A tabela pública mede especialização nessas tarefas; não isola idioma de domínio |
| Dataset traduzido PT, anúncios e documentos internos | Receitas e cards | Não usar esses domínios como evidência automática de novidade; conferir itens, grupos e derivados |
| Perguntas geradas sobre textos reais | Receitas e cards | Trocar a pergunta não torna o texto independente |
| Oito casos sintéticos, incluindo e-mail e tickets | Ambos os cards fixados | Os pilotos propostos testam novos textos/operações, não tarefas inteiramente inéditas |
| Avaliações humanas antigas de e-mail e comentários | Cards declaram calibração sobre essas avaliações | Reutilizar como desenvolvimento/diagnóstico, não como prova final intocada |

O 322M deriva do Laya multilíngue. O 4B declara inicialização no Kev 4B sobre Qwen3.5-4B-Base. A edição privada ajustada com e-mails e o candidato privado de retreino são checkpoints diferentes e não participam deste piloto. O parente selecionado na rodada é uma referência observada, não prova adicional da revisão exata de inicialização histórica.

O código separa os splits públicos de treino e teste; isso não prova a composição do arquivo materializado de um treino antigo nem ausência de duplicatas entre fontes. Os arquivos de treino, itens tokenizados e recibos por item não estão no checkout inspecionado. O carregamento do dataset traduzido nas receitas não fixa revisão, e coleções internas entram por globs opcionais. Por isso não foi possível reconciliar integralmente contagens, versões e exposições de cada execução. Isso é uma lacuna de procedência, não evidência de vazamento do teste.

## Material reaproveitado e fontes procuradas

Reaproveitamos as perguntas históricas de triagem, o contrato System One, os seis checkpoints fixados, o embaralhamento de opções e as métricas do avaliador. Os textos e rótulos antigos não são promovidos a dados novos. A busca em arquivos de projetos disponíveis e suas referências não localizou um corpus real elegível de atendimento; encontrou código, logs de infraestrutura e evidências operacionais, que não substituem conversas reais. A busca não consultou bases de produção, caixas de e-mail ou Drives e não prova que não exista corpus nesses sistemas.

A fonte pública preferida para o piloto de atendimento é [ConsumerBR](https://zenodo.org/records/18022568), documentada no [artigo dos autores](https://aclanthology.org/2026.propor-1.66/). São reclamações reais anonimizadas e respostas empresariais. O registro usa CC BY 4.0 no campo de licença, mas a descrição e o artigo restringem uso à pesquisa: manter essa divergência explícita, não redistribuir textos ou anunciar licença irrestrita. O arquivo tem cerca de 1,64 GB comprimido; não foi baixado integralmente. O período é anterior aos checkpoints: não sustenta novidade temporal por si só.

O dataset Brazilian Customer Service Conversations foi rejeitado como atendimento real porque os registros inspecionados declaram geração sintética. TalkEx deriva dessa fonte: marcar registros como não aumentados não comprova origem humana. MilkQA contém perguntas reais, mas corresponde a seleção de respostas e tem restrições de licença; não substitui automaticamente triagem de tickets. A Busca LAI da CGU é alternativa para solicitações escritas públicas, sujeita a inspeção do export; não será chamada de e-mail empresarial. Detalhes e URLs no [inventário de fontes](../../research/rounds/2026-10-31-transfer-pilots/source-discovery.json).

## Pilotos preparados

| | Piloto 1 | Piloto 2 |
|---|---|---|
| Caso | Triagem de e-mails/solicitações empresariais | Triagem e roteamento de atendimento |
| Perguntas | Tipo, exige ação, urgência | Motivo, encaminhamento, escalonamento |
| Reuso | Definição histórica de perguntas, generalizada para a organização receptora | Esquema inicial explícito, a revisar sem respostas dos modelos |
| Fonte | Novos e-mails independentes ainda não localizados; alternativa de solicitações públicas exige outro nome e taxonomia | ConsumerBR como candidato público, sem depender de corpus interno |
| Quantidade | 50 itens de desenvolvimento, depois alvo inicial de 250 itens finais novos | 50 itens de desenvolvimento, depois alvo inicial de 250 itens finais novos |
| Estado atual | Protocolo pronto; zero itens independentes verificados | Protocolo pronto; zero itens independentes verificados |

Os seis modelos do piloto são Saracura v0, Saracura 4B, Laya multilingual, Kev 4B, Lev e Qwen3.5-4B, nas revisões da rodada publicada. Não adaptar pesos ou calibrar confiança com o teste final. A rodada confirmatória deve incluir os 14, sem escolher comparadores por seus resultados no piloto. O piloto serve para acertar rubrica e viabilidade, não para anunciar vencedor.

## Critérios para aceitar dados

1. Registrar snapshot, licença/condições, origem humana ou sintética, período e identificador de grupo. Conservar texto e hashes sensíveis em armazenamento privado autorizado, fora do Git público.
2. Reconstruir índices de exclusão de treino, desenvolvimento, calibração e avaliações já usadas para selecionar modelos, incluindo dados gerados e seus textos de origem. Se o índice estiver incompleto, o item fica em quarentena: ausência de match em um índice parcial não prova independência.
3. Verificar duplicação exata com normalização NFC e espaços; revisar quase duplicatas, templates, traduções e variantes. Preservar parentesco por conversa, consumidor/empresa, ticket ou documento. Nenhum grupo atravessa desenvolvimento e teste. Identificadores públicos não são fornecidos aos modelos como pistas.
4. Para corpus público anterior ao checkpoint, exigir confronto com os insumos próprios; declarar exposição upstream desconhecida. Para corpora internos ausentes, aguardar cópia autorizada fora da VM, sem depender de ligar a VM nesta etapa. Não usar a expressão livre de contaminação.
5. Dois anotadores independentes, sem respostas de modelos, resolvem divergências com adjudicação. Ausência de informação é anotada e revisada; não se inventam prazos ou políticas. Não excluir exemplos porque um modelo errou.
6. Fornecer a todos o mesmo contexto empresarial, referência temporal e mapa de responsabilidades. Entram somente mensagens disponíveis no momento da triagem. Respostas futuras da empresa, satisfação, resolução e rótulos ficam fora do input. Encaminhamento no piloto público mede conformidade com política explícita; não se apresenta uma equipe hipotética como encaminhamento observado.
7. Congelar perguntas, dados, splits, critérios de contexto e métricas antes de inferência. Não acrescentar corte silencioso de texto; preflight comum documenta comprimento e limites nativos. Formato inválido e ausência de resposta contam como falha. A amostra de desenvolvimento nunca volta como teste final.

## Métricas e publicação

Reportar balanced accuracy por pergunta e por caso, macro-F1, erros de ação/prioridade/escalonamento, taxa de resposta válida e concordância humana antes de adjudicação. Manter resultados pareados por item; intervalos por bootstrap agrupado na conversa/documento e diferenças pareadas, sem tratar perguntas do mesmo texto como observações independentes. Metas de erros críticos são fixadas com o responsável operacional após o piloto e antes da prova final; não são inventadas como fatos do negócio.

A correlação entre a tabela pública e os casos reais, além da perda ao escolher seu líder em vez do melhor modelo de cada caso, será análise exploratória na rodada dos 14. Latência e memória só entram em execução futura controlada no mesmo hardware. Não haverá custo, latência ou confiança calibrada fictícios nesta preparação.

Publicar os resultados futuros em trilha separada de generalização; preservar a tabela atual como benchmark público com exposição às tarefas identificada. Adaptação com orçamento comum e a investigação causal idioma versus domínio são etapas posteriores. Um resultado inferior do Saracura não altera critérios, seleção ou publicação.

## Próxima ação concreta

Resolver a fonte de textos do piloto 1 e as condições do ConsumerBR; obter amostras pequenas em quarentena, revisar campos e formar o índice de exclusão. Só depois selecionar os 50 itens por caso e encaminhar à anotação. Até esse ponto, nenhum dos dois pilotos está liberado para inferência. Isso preserva a exigência de independência sem transformar ausência de dados em uma aprovação automática.
