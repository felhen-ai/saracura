"""Servidor /v1/systemone mínimo para modelos que só expõem uma classe Python.

Uso: ENGINE=intern MODEL_DIR=~/board/models/intern-4b uvicorn pyserve:app --port 8030
     ENGINE=jet    MODEL_DIR=~/board/models/jet          uvicorn pyserve:app --port 8031
"""

import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ENGINE = os.environ["ENGINE"]
MODEL_DIR = os.path.expanduser(os.environ["MODEL_DIR"])
sys.path.insert(0, MODEL_DIR)
os.chdir(MODEL_DIR)

if ENGINE == "intern":
    from inference import DecisionEngine

    _engine = DecisionEngine(device="cuda")

    def decide(req):
        return _engine.predict({"state": req["state"], "questions": req["questions"]})
elif ENGINE == "jet":
    from jet import Jet

    _engine = Jet()

    def decide(req):
        return _engine.decide(req["state"], req["questions"])
else:
    raise SystemExit(f"ENGINE desconhecido: {ENGINE}")

app = FastAPI()


@app.get("/docs-ok")
def ok():
    return {"ok": True}


@app.post("/v1/systemone")
async def systemone(request: Request):
    req = await request.json()
    try:
        out = decide(req)
    except Exception as e:  # erro volta ao avaliador e conta como falha
        return JSONResponse({"error": type(e).__name__, "detail": str(e)[:300]}, status_code=422)
    if isinstance(out, dict) and "answers" in out:
        return out
    return {"answers": out}
