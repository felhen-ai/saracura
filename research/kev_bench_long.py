"""kev.benchmark com o contexto de treino (estado até 2048 tokens) em vez do padrão de 384, para dados próprios."""

import sys

import kev.benchmark as b
from kev.model import training_context

b.CONTEXT = {**training_context(2048), "truncate": False}
sys.exit(b.main())
