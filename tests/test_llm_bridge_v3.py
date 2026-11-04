import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_research_module(name: str) -> Any:
    path = Path(__file__).parents[1] / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


llm_bridge = _load_research_module("llm_bridge")
v3 = _load_research_module("llm_bridge_v3")


def test_case_sensitive_32_identifier_readout_and_legacy_is_unchanged() -> None:
    assert v3.parse_identifier("f", 32) == 31
    assert v3.parse_identifier("F", 32) == 5
    assert v3.parse_identifier("f", 31) is None
    assert v3.parse_identifier("ff", 32) is None
    assert llm_bridge.parse_letter("f", 26) is None
    assert llm_bridge.LETTERS == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_v3_prompt_hash_and_render_are_distinct_from_legacy() -> None:
    q = llm_bridge.Question(
        "q", "choice", "Escolha", tuple(str(i) for i in range(32)), tuple(str(i) for i in range(32))
    )
    rendered = v3.render_user_message("estado", q)
    assert "f) 31" in rendered
    assert "sensível" in rendered
    assert v3.PROMPT_SHA256 != llm_bridge.PROMPT_SHA256


def test_v3_accepts_32_choices_without_changing_legacy_limit() -> None:
    criteria = {str(i): str(i) for i in range(32)}
    raw = json.dumps(
        {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "i", "criteria": criteria}},
        }
    ).encode()
    assert len(v3.parse_request(raw, "m").questions[0].keys) == 32
    try:
        llm_bridge.parse_request(raw, "m")
    except llm_bridge.RequestError:
        pass
    else:
        raise AssertionError("legacy protocol must retain its 26-choice maximum")
