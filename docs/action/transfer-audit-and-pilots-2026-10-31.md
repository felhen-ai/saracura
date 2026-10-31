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

Auditoria documental concluída para os dois checkpoints públicos exatos da rodada dos 14. Foram examinados os model cards nas revisões fixadas, as receitas de preparação, as definições das provas privadas e a disponibilidade dos insumos no checkout local. Não houve acesso à VM 151, carregamento de modelos, contratação de GPU, treino ou inferência. O resultado é um mapa de exposição, dois protocolos e 100 textos candidatos de desenvolvimento preparados; nenhum item foi certificado como plenamente independente e nenhuma nova pontuação foi produzida.

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

A fonte pública preferida para o piloto de atendimento é [ConsumerBR](https://zenodo.org/records/18022568), documentada no [artigo dos autores](https://aclanthology.org/2026.propor-1.66/). São reclamações reais anonimizadas e respostas empresariais. O registro usa CC BY 4.0 no campo de licença, mas a descrição e o artigo restringem uso à pesquisa: manter essa divergência explícita, não redistribuir textos ou anunciar licença irrestrita. O arquivo tem cerca de 1,64 GB comprimido; foram baixados somente os primeiros 2 MiB, sem extração integral. O período é anterior aos checkpoints: não sustenta novidade temporal por si só.

O dataset Brazilian Customer Service Conversations foi rejeitado como atendimento real porque os registros inspecionados declaram geração sintética. TalkEx deriva dessa fonte: marcar registros como não aumentados não comprova origem humana. MilkQA contém perguntas reais, mas corresponde a seleção de respostas e tem restrições de licença; não substitui automaticamente triagem de tickets. A Busca LAI da CGU é alternativa para solicitações escritas públicas, sujeita a inspeção do export; não será chamada de e-mail empresarial. Detalhes e URLs no [inventário de fontes](../../research/rounds/2026-10-31-transfer-pilots/source-discovery.json).

## Pilotos preparados

| | Piloto 1 | Piloto 2 |
|---|---|---|
| Caso | Triagem de solicitações públicas escritas | Triagem e roteamento de atendimento |
| Perguntas | Objetivo, tema, precisa esclarecimento | Motivo, encaminhamento, escalonamento |
| Reuso | Contrato de decisão e metodologia; taxonomia própria para pedidos públicos | Esquema inicial explícito, a revisar sem respostas dos modelos |
| Fonte | Export oficial CGU2026; 50 pedidos registrados de 22 a 29/10, depois do congelamento em 21/10 | ConsumerBR; 50 reclamações reais de desenvolvimento |
| Quantidade | 50 textos preparados; alvo futuro de 250 itens finais novos | 50 textos preparados; alvo futuro de 250 itens finais novos |
| Estado atual | Sem rótulos humanos; revisão de privacidade e índice completo pendentes | Sem rótulos humanos; revisão de privacidade e índice completo pendentes |

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

Revisar privacidade e completar a exclusão dos insumos privados/gerados; depois encaminhar as duas fichas separadas por anotador e adjudicar divergências. Os 50 itens por caso já estão preparados, mas ambos os conjuntos permanecem em quarentena para inferência e não constituem prova final. Não há dependência da VM 151 nesta preparação.


## Amostras preparadas e verificações em 31/10

O usuário autorizou o ConsumerBR especificamente para pesquisa. Aplicamos as condições mais restritivas declaradas: release anonimizado, citação, proibição de reidentificação, sem redistribuição pública dos textos e sem uso operacional ou treino. A inconsistência entre CC BY 4.0 no campo de licença e a descrição research-only continua registrada; não foi reinterpretada como licença irrestrita. A documentação da CGU tem rodapé CC BY-ND 3.0, que não prova direitos irrestritos sobre narrativas de terceiros: os textos também ficam apenas no acervo restrito de pesquisa.

ConsumerBR: 50 textos sorteados com semente 20261031 entre 981 narrativas únicas das primeiras 1.000 linhas completas de um prefixo comprimido limitado. Todos são de 13/09/2021. Essa amostra serve para depurar a rubrica; não representa o corpus inteiro nem sustenta novidade temporal. A seleção final precisará de cobertura temporal e de empresas mais ampla. Foram excluídas do input respostas das empresas, satisfação e resolução.

CGU: o snapshot de 30/10 contém 67.506 pedidos de2026. O filtro de registro posterior a21/10 encontrou 260 textos não vazios únicos; uma amostragem reservoir com semente 20261031 escolheu 50, datados de 22 a 29/10. A fonte é real e posterior ao congelamento dos checkpoints fixados, mas a reutilização de histórias antigas não é descartada só pela data. O input contém apenas resumo e detalhamento; resposta, assunto atribuído pelo SIC, órgão destinatário e decisão não entram como pistas. Duas amostras têm mais de 3.500 caracteres e foram preservadas, com pendência explícita de contexto; não houve corte silencioso.

O piloto1 foi renomeado: não há corpus novo de e-mail empresarial comprovado. Removemos a pergunta exige ação porque um acervo composto só de pedidos não permite presumir uma classe negativa representativa. A alternativa mede objetivo, tema e necessidade de esclarecimento; não é comparação direta com os 59,9% históricos de e-mail/comentários.

Conferimos os 100 candidatos contra 39.295 registros dos 21 arquivos públicos train/dev/test na revisão 00f5d6dcc6679751723289889b8b0b28b71cbcb9: zero duplicatas exatas normalizadas, incluindo comparação com estados de treino cortados em 600/1.500/3.500 caracteres. Dentro dos 100 candidatos, zero duplicatas exatas e zero pares acima de Jaccard 0,8 em trigramas de palavras. Isso não cobre similaridade semântica, traduções, insumos internos, perguntas geradas, calibração ou pré-treino upstream. O veredito continua independência parcial, não certificação de ausência de contaminação.

Há quatro fichas de anotação vazias, uma por corpus/anotador, sem respostas de modelos ou gabarito automático. Redação por regex é preliminar; revisar nomes e outros identificadores antes de disponibilizar a anotadores. As amostras, snapshots, scripts de execução capturados e recibos de pesquisa ficam no armazenamento restrito, caminho lógico projetos/felhen/saracura/.artifacts/transfer-pilots-research-20261031. O repositório contém apenas protocolo, metadados agregados e hashes de artefatos. O [recibo público de preparação](../../research/rounds/2026-10-31-transfer-pilots/sample-preparation.json) registra escopo, amostragem e pendências.


## Revisão de privacidade e extensão da sobreposição

A revisão assistida examinou os 100 estados candidatos e aplicou redações determinísticas adicionais a nomes, assinaturas, contatos, endereços, placas, reservas, logins e referências longas. Foram separados 16 estados por saúde/deficiência/gravidez, raça/etnia, menores/família, registros pessoais sensíveis ou contexto individual identificável. Não houve substituição baseada em respostas de modelos; os originais foram preservados como histórico em quarentena. Restaram 47 textos ConsumerBR e 37 pedidos CGU em privacy-reviewed-v2, com quatro novas fichas vazias. A revisão é assistida por modelo, não anotação humana dupla ou certificação de anonimização irreversível. Um pedido ainda requer resolução do limite de contexto antes de inferência. As fichas anteriores não devem ser usadas para anotação.

A busca no projeto do Shared Drive restrito listou 132 arquivos, sem os insumos materializados procurados. Uma busca posterior por nomes exatos nos metadados corporativos encontrou balanced_training_25k.jsonl em outra pasta; ele foi recuperado em modo somente leitura, com 24.999 registros, e usado integralmente como conjunto conservador de exclusão. A cópia de trabalho ficou fora do Git e fora da entrega de amostras para evitar duplicar textos privados. O arquivo de origem existente no Drive continua preservado. Não se afirma que essa cópia é exatamente a versão ou o subconjunto usado no treino antigo.

Também foram conferidos os 1.200 registros de treino PT do dataset traduzido disponível, revisão 8c85a4d2017c12599857ad083f87536fa6424596. A receita histórica não fixa essa revisão; portanto esta é uma checagem adicional, não reconstrução comprovada da execução antiga. Nos 26.199 registros adicionais, houve zero matches exatos e zero pares com Jaccard de trigramas maior ou igual a 0,8 contra os 100 estados originais, antes das redações de privacidade. A checagem exata inclui estados completos e prefixos de 600/1.500/2.500/3.500 caracteres. O prefixo de 2.500 é usado pelos geradores de perguntas sobre textos reais nas receitas inspecionadas. Somados os 39.295 registros públicos, foram comparados 65.494 registros; essa contagem não implica 65.494 textos distintos.

Os arquivos originais de casos sintéticos e splits de documentos internos não foram localizados nas buscas de nomes e no acervo inspecionado dos Shared Drives corporativos e do Mac. Não houve acesso à VM 151, aos bancos de produção ou a caixas de e-mail. Permanecem faltando os datasets gerados efetivamente usados, os splits internos e os índices de exposição de calibração/seleção; similaridade semântica/tradução e pré-treino upstream não estão certificados. As receitas dos geradores foram inspecionadas sem executá-las ou chamar professores.

Por isso a execução autônoma dos modelos não foi liberada: o índice de exclusão ainda é incompleto e há zero gabaritos humanos. Produzir rótulos com o próprio assistente e chamá-los de validação humana ou acurácia resolveria artificialmente a lacuna; não foi feito. Não há modelo rodando em background. A automação existente acompanha somente os checks e a integração desta documentação. O [recibo de privacidade e sobreposição](../../research/rounds/2026-10-31-transfer-pilots/privacy-and-overlap-review.json) registra as contagens e os bloqueios de execução.
