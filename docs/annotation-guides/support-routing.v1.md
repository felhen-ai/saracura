---
title: Support routing v1 annotation guide
kind: guide
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/annotation-guides/support-routing.v1.md
globalRef: qmd://saracura/docs/annotation-guides/support-routing.v1.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs: []
related:
  - benchmarks/manifests/support-routing-protocol.v1.json
  - docs/action/specs/phase3a-first-party-packet.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Support routing v1

Este guia congela uma única decisão: qual fila é dona da próxima ação
material em uma solicitação escrita em português brasileiro. A anotação é
single-label. Menções a outros assuntos não criam multilabel.

## Filas e prioridade

Aplicar as regras nesta ordem:

1. `account_access`: autenticação, verificação de identidade, recuperação de
   credencial ou alcance da conta bloqueiam todas as demais ações pedidas.
2. `billing`: cobrança concluída ou tentada, pagamento, estorno, fatura ou
   ação sobre método de pagamento, exceto quando a única ação pedida é cancelar.
3. `subscription_cancellation`: parar assinatura, renovação ou plano
   recorrente quando não há cobrança ou estorno separado já concluído.
4. `order_delivery`: envio, rastreio, entrega, pacote ausente, entrega danificada
   ou cumprimento de pedido físico.
5. `technical_support`: falha, configuração, compatibilidade ou ajuda de uso
   que não esteja bloqueada por acesso à conta.

`ambiguous` é usado quando permanece igualdade de propriedade, falta contexto
obrigatório ou uma revisão independente futura não consegue resolver o rótulo
pelas regras congeladas. `out_of_scope` é usado quando nenhuma fila possui a
próxima ação. `rejected` é um resultado de revisão, não uma classe de decisão.

## Limites de autoria, privacidade e direitos

O pacote futuro aceita apenas texto original escrito por uma pessoa autorizada,
com `author_id` opaco e atestado externo verificável pelos mantenedores. Dois
agentes, duas contas de uma pessoa ou uma pessoa em papéis diferentes não
provam independência. O validador não prova que a declaração é verdadeira nem
que o texto está livre de informação pessoal.

O autor declara `no_personal_data` e `approved_first_party`; a ferramenta faz
apenas uma prechecagem mecânica deliberadamente conservadora. Privacidade,
direitos, autoria e revisão independente permanecem gates externos. Textos
gerados, assistidos por modelo, de clientes, traces privados ou texto público
não vetado não entram nesta lane.

## Famílias e holdout

Cada família tem exatamente um estado no inventário desta versão. A atribuição
é feita antes de qualquer variante ou derivativo e todas as variantes futuras
devem permanecer na mesma família. `train`, `dev`, `calibration` e `blind_test`
são grupos imutáveis após o plano ser congelado. Acesso ao blind test é
one-shot, por operador autorizado, depois de revisão; o plano não autoriza
treino, publicação, qualidade ou automação.

Trechos abaixo, se adicionados em documentação, são somente ilustrativos: não
têm IDs, não são registros, não são evidência de autoria humana e nunca são
importados pelo código.
