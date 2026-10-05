"""Textos sintéticos de casos de uso reais (e-mail, ticket, lead, WhatsApp, documentos, avaliações, moderação, agente)
com perguntas canônicas e livres, gerados e respondidos por um professor grande de pesos abertos via OpenRouter.

Duas chamadas por texto: (1) escrever o texto a partir de um brief sorteado (persona, tom, tamanho, valores-alvo das
perguntas canônicas, para balancear classes); (2) responder às perguntas canônicas às cegas e criar/responder 2 perguntas
livres. Texto cujas respostas às cegas divergem muito do brief é descartado (texto ambíguo ou mal escrito).

  python gen_usecases.py --name pilot_uc --per-case 6 --budget 2
  python gen_usecases.py --name train_uc --per-case 625 --budget 30
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gen_openrouter import MODEL, Budget, chat, parse_json, render_questions, to_gold, to_question

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "data" / "gen_uc"
SEED = 20261004

CASES = {
    "email": {
        "texto": "um e-mail recebido por uma empresa brasileira (com assunto e corpo)",
        "canonicas": {
            "tipo": {
                "type": "choice",
                "instructions": "Que tipo de e-mail é este?",
                "criteria": {
                    "lead": "contato comercial de alguém interessado em comprar ou contratar",
                    "campanha": "marketing, newsletter ou promoção enviada em massa",
                    "suporte": "pedido de ajuda ou reclamação de cliente",
                    "cobranca": "fatura, boleto, pagamento ou cobrança",
                    "interno": "comunicação entre colegas da própria empresa",
                    "spam": "golpe, phishing ou mensagem indesejada",
                },
            },
            "prioridade": {
                "type": "score",
                "instructions": "Qual é a prioridade de resposta?",
                "criteria": ["baixa", "média", "alta", "urgente"],
            },
            "precisa_resposta": {
                "type": "noul",
                "instructions": "Este e-mail exige uma resposta humana?",
            },
        },
    },
    "ticket": {
        "texto": "um chamado de suporte aberto por um cliente de um produto ou serviço brasileiro",
        "canonicas": {
            "equipe": {
                "type": "choice",
                "instructions": "Qual equipe deve atender este chamado?",
                "criteria": {
                    "financeiro": "cobranças, pagamentos, estornos e notas fiscais",
                    "tecnico": "falhas, erros e indisponibilidade",
                    "comercial": "planos, upgrades, cancelamento e renegociação",
                    "logistica": "entrega, prazo, extravio e devolução",
                    "cadastro": "acesso, senha, dados cadastrais e permissões",
                },
            },
            "urgencia": {
                "type": "score",
                "instructions": "Qual é a urgência do chamado?",
                "criteria": ["baixa", "média", "alta", "crítica"],
            },
            "cliente_irritado": {
                "type": "noul",
                "instructions": "O cliente demonstra irritação ou ameaça cancelar?",
            },
        },
    },
    "lead": {
        "texto": "uma mensagem ou formulário de contato enviado a uma empresa B2B brasileira por um possível cliente",
        "canonicas": {
            "estagio": {
                "type": "choice",
                "instructions": "Em que estágio de compra está este contato?",
                "criteria": {
                    "curioso": "só quer entender o que a empresa faz",
                    "pesquisando": "compara opções e pede informações",
                    "pronto": "quer proposta, preço ou reunião",
                    "nao_lead": "não é um potencial cliente (fornecedor, candidato, estudante)",
                },
            },
            "porte": {
                "type": "choice",
                "instructions": "Qual o porte provável da empresa do contato?",
                "criteria": {
                    "pessoa_fisica": "pessoa física ou autônomo",
                    "pequena": "pequena empresa",
                    "media": "média empresa",
                    "grande": "grande empresa ou governo",
                    "indefinido": "não dá para saber",
                },
            },
            "pede_orcamento": {
                "type": "noul",
                "instructions": "O contato pede explicitamente preço, orçamento ou proposta?",
            },
        },
    },
    "whatsapp": {
        "texto": "uma mensagem curta de cliente enviada pelo WhatsApp a um negócio brasileiro (clínica, loja, escola, prestador de serviço)",
        "canonicas": {
            "intencao": {
                "type": "choice",
                "instructions": "Qual é a intenção da mensagem?",
                "criteria": {
                    "agendar": "marcar, remarcar ou cancelar horário",
                    "preco": "perguntar preço ou condições",
                    "status": "saber andamento de pedido, exame ou solicitação",
                    "reclamar": "reclamar ou relatar problema",
                    "saudacao": "cumprimento ou mensagem sem pedido claro",
                    "outro": "outro assunto",
                },
            },
            "precisa_humano": {
                "type": "noul",
                "instructions": "Esta mensagem precisa de atendimento humano, em vez de resposta automática?",
            },
            "sentimento": {
                "type": "score",
                "instructions": "Qual é o sentimento da mensagem?",
                "criteria": ["negativo", "neutro", "positivo"],
            },
        },
    },
    "documento": {
        "texto": "o texto extraído de um documento financeiro ou administrativo brasileiro (fatura, boleto, nota fiscal, contrato, recibo, ofício)",
        "canonicas": {
            "tipo_doc": {
                "type": "choice",
                "instructions": "Que tipo de documento é este?",
                "criteria": {
                    "fatura": "fatura ou boleto de cobrança",
                    "nota_fiscal": "nota fiscal ou cupom",
                    "contrato": "contrato ou termo",
                    "recibo": "recibo ou comprovante de pagamento",
                    "oficio": "ofício, notificação ou comunicado formal",
                },
            },
            "area": {
                "type": "choice",
                "instructions": "Qual área deve tratar este documento?",
                "criteria": {
                    "contas_a_pagar": "pagamentos a fornecedores",
                    "contas_a_receber": "recebimentos de clientes",
                    "juridico": "contratos e questões legais",
                    "rh": "pessoal e folha",
                    "fiscal": "impostos e obrigações fiscais",
                },
            },
            "vencido": {
                "type": "noul",
                "instructions": "O documento indica um prazo ou pagamento já vencido?",
            },
        },
    },
    "avaliacao": {
        "texto": "uma avaliação escrita por um consumidor brasileiro sobre um produto comprado em marketplace",
        "canonicas": {
            "nota": {
                "type": "score",
                "instructions": "Qual nota de 1 a 5 esta avaliação sugere?",
                "criteria": ["1", "2", "3", "4", "5"],
            },
            "problema": {
                "type": "choice",
                "instructions": "Qual é o principal problema relatado, se houver?",
                "criteria": {
                    "nenhum": "nenhum problema",
                    "defeito": "produto com defeito ou quebrado",
                    "entrega": "atraso, extravio ou embalagem",
                    "diferente": "produto diferente do anunciado",
                    "atendimento": "atendimento ou pós-venda ruim",
                    "preco": "preço ou custo-benefício",
                },
            },
            "recomenda": {"type": "noul", "instructions": "O consumidor recomendaria o produto?"},
        },
    },
    "moderacao": {
        "texto": "um comentário publicado por um usuário em uma rede social ou site de notícias brasileiro",
        "canonicas": {
            "acao": {
                "type": "choice",
                "instructions": "Que ação a moderação deve tomar?",
                "criteria": {
                    "manter": "comentário aceitável",
                    "revisar": "caso duvidoso, revisar manualmente",
                    "remover": "viola as regras e deve ser removido",
                },
            },
            "ofensivo": {"type": "noul", "instructions": "O comentário contém linguagem ofensiva?"},
            "alvo": {
                "type": "choice",
                "instructions": "A quem se dirige o comentário?",
                "criteria": {
                    "ninguem": "não ataca ninguém",
                    "individuo": "uma pessoa específica",
                    "grupo": "um grupo de pessoas",
                    "instituicao": "uma empresa, governo ou instituição",
                },
            },
        },
    },
    "agente": {
        "texto": "o registro resumido de uma ação proposta por um agente de IA em um sistema de uma empresa (tarefa, ferramenta, parâmetros e justificativa), em português",
        "canonicas": {
            "decisao": {
                "type": "choice",
                "instructions": "O que o supervisor deve fazer com esta ação?",
                "criteria": {
                    "continuar": "deixar o agente prosseguir",
                    "observar": "prosseguir, mas marcar para auditoria",
                    "revisar": "pausar e pedir revisão humana",
                    "parar": "interromper o agente agora",
                },
            },
            "risco": {
                "type": "score",
                "instructions": "Qual é o risco da ação?",
                "criteria": ["baixo", "médio", "alto", "crítico"],
            },
            "irreversivel": {
                "type": "noul",
                "instructions": "A ação é irreversível (apaga, envia, paga ou publica)?",
            },
        },
    },
}

STYLES = [
    "formal",
    "coloquial",
    "muito informal, com gírias",
    "apressado, com erros de digitação e sem acentos",
    "educado e detalhado",
    "seco e direto",
    "confuso e prolixo",
    "em caixa alta em partes",
    "com emojis",
    "regionalista (Nordeste)",
    "regionalista (Sul)",
    "corporativo",
]
SIZES = [
    "muito curto (1 ou 2 frases)",
    "curto (um parágrafo)",
    "médio (2 a 3 parágrafos)",
    "longo (4 ou mais parágrafos)",
]

WRITE_PROMPT = """Escreva {texto}, em português do Brasil, realista e específico (nomes, valores, datas e produtos plausíveis e variados; nunca use placeholders como [nome]).
Estilo: {style}. Tamanho: {size}.
O texto deve ser consistente com estas características, sem citá-las explicitamente:
{targets}

Responda só com JSON: {{"texto": "..."}}"""

ANSWER_PROMPT = """Leia o texto e faça duas coisas.
1) Responda às perguntas canônicas distribuindo 100% de probabilidade entre as opções (concentre quando o texto for claro).
2) Crie 2 perguntas novas e úteis sobre este texto, de tipos diferentes das canônicas quando possível ("escolha" com "opcoes" {{id: descrição}}, "sim_nao", ou "escala" com "niveis" do menor ao maior), e responda-as da mesma forma.

Texto:
\"\"\"{text}\"\"\"

Perguntas canônicas:
{questions}

Responda só com JSON: {{"respostas": {{"<id>": {{"<id_opcao>": prob, ...}}}}, "novas": [{{"tipo": ..., "instrucao": ..., "opcoes": {{...}} ou "niveis": [...], "resposta": {{"<id_opcao>": prob, ...}}}}]}}"""


def target_text(q, value):
    if q["type"] == "choice":
        return f"{q['instructions']} -> {q['criteria'][value]}"
    if q["type"] == "noul":
        return f"{q['instructions']} -> {'sim' if value else 'não'}"
    return f"{q['instructions']} -> {q['criteria'][value]}"


def sample_targets(case, rng):
    targets = {}
    for qid, q in case["canonicas"].items():
        if q["type"] == "choice":
            targets[qid] = rng.choice(list(q["criteria"]))
        elif q["type"] == "noul":
            targets[qid] = rng.random() < 0.5
        else:
            targets[qid] = rng.randrange(len(q["criteria"]))
    return targets


def agrees(q, target, gold):
    probs = gold["probabilities"]
    top = max(probs, key=probs.get)
    if q["type"] == "noul":
        return (top == "true") == target
    if q["type"] == "score":
        return abs(int(top) - target) <= 1
    return top == target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-case", type=int, default=625)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY ausente no ambiente")
    budget = Budget(args.budget)
    rng = random.Random(args.seed)
    jobs = [
        (name, sample_targets(case, rng), rng.choice(STYLES), rng.choice(SIZES))
        for name, case in CASES.items()
        for _ in range(args.per_case)
    ]
    rng.shuffle(jobs)
    print(
        f"modelo={MODEL} textos={len(jobs)} casos={len(CASES)} teto=US$ {args.budget}", flush=True
    )

    def work(job):
        name, targets, style, size = job
        case = CASES[name]
        tg = "\n".join(f"- {target_text(q, targets[qid])}" for qid, q in case["canonicas"].items())
        w = parse_json(
            chat(
                key,
                budget,
                WRITE_PROMPT.format(texto=case["texto"], style=style, size=size, targets=tg),
                900,
                0.9,
            )
        )
        text = (w or {}).get("texto")
        if not isinstance(text, str) or len(text) < 40:
            return None
        canon = {qid: dict(q) for qid, q in case["canonicas"].items()}
        a = parse_json(
            chat(
                key,
                budget,
                ANSWER_PROMPT.format(text=text, questions=render_questions(canon)),
                900,
                0.0,
            )
        )
        if not a:
            return None
        gold, questions = {}, {}
        resp = a.get("respostas") or {}
        for qid, q in canon.items():
            g = to_gold(q, resp.get(qid))
            if g:
                questions[qid], gold[qid] = q, g
        if (
            len(gold) < 2
            or sum(agrees(canon[qid], targets[qid], gold[qid]) for qid in gold) < len(gold) - 1
        ):
            return None  # texto ambíguo: o professor às cegas discorda do brief em mais de uma pergunta
        for i, p in enumerate((a.get("novas") or [])[:2]):
            q = to_question(p)
            g = to_gold(q, p.get("resposta")) if q else None
            if q and g:
                questions[f"n{i}"], gold[f"n{i}"] = q, g
        return {
            "source": f"uc_{name}",
            "state": text,
            "questions": questions,
            "gold": gold,
            "teacher": MODEL,
            "brief": {"style": style, "size": size},
        }

    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / f"{args.name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["state"] for line in out.open()}
    t0, n_rows, n_q, n_drop = time.time(), 0, 0, 0
    try:
        with ThreadPoolExecutor(args.workers) as ex, out.open("a") as f:
            for i, row in enumerate(ex.map(work, jobs[len(done) :])):
                if row:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n_rows += 1
                    n_q += len(row["questions"])
                else:
                    n_drop += 1
                if (i + 1) % 100 == 0:
                    print(
                        f"{i + 1}/{len(jobs)} briefs, {n_rows} textos, {n_q} perguntas, {n_drop} descartados, US$ {budget.spent:.2f}, {round(time.time() - t0)} s",
                        flush=True,
                    )
    finally:
        src = Counter(json.loads(line)["source"] for line in out.open())
        print(
            f"gravado {out}: {n_rows} textos novos, {n_q} perguntas, {n_drop} descartados, por caso={dict(src)}, US$ {budget.spent:.2f} em {budget.calls} chamadas"
        )


if __name__ == "__main__":
    main()
