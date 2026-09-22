from __future__ import annotations

from collections.abc import Sequence

from benchmarks.universal_local.inputs import (
    TextChoice,
    TextState,
    frame_state,
    render_laya,
    render_nli,
)


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103
    all_special_tokens: Sequence[str] = ("[CLS]", "[SEP]", "[MASK]")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(10, 10 + len(text.encode())))

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int:
        return 3 if pair else 2

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is None:
            return [self.cls_token_id, *token_ids_0, self.sep_token_id]
        return [self.cls_token_id, *token_ids_0, self.sep_token_id, *token_ids_1, self.sep_token_id]


class NonCompositionalTokenizer(FakeTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        raw = text.encode()
        return [len(raw), sum(raw) % 997]


def test_state_frame_is_utf8_length_framed() -> None:
    assert frame_state(TextState("á", "ok")) == "S2:áB2:ok"


def test_laya_renderer_inserts_markers_without_choice_ids() -> None:
    rendered = render_laya(
        FakeTokenizer(),
        TextState("assunto", "corpo suficiente"),
        "instrucao",
        [TextChoice("private-id", "descricao")],
    )
    assert rendered.status == "supported"
    assert rendered.input_ids[rendered.marker_positions[0]] == 103
    assert "private-id" not in str(rendered.input_ids)


def test_renderer_rejects_special_literal_and_nli_has_locale_assertion() -> None:
    assert (
        render_laya(
            FakeTokenizer(), TextState("[MASK]", "body"), "instruction", [TextChoice("x", "choice")]
        ).status
        == "unsupported_input"
    )
    rendered = render_nli(
        FakeTokenizer(),
        TextState("subject", "body"),
        "instruction",
        TextChoice("x", "choice"),
        "en",
    )
    assert rendered.status == "supported"
    assert rendered.input_ids[0] == 101
    assert rendered.input_ids.count(102) == 2
    assert rendered.pair_positions == (1, 18, 19, 87)
    assert rendered.hypothesis.endswith("The description above is the correct option.")


def test_nli_tokenizes_the_exact_canonical_premise_in_one_call() -> None:
    tokenizer = NonCompositionalTokenizer()
    rendered = render_nli(
        tokenizer,
        TextState("assunto", "corpo"),
        "instrução",
        TextChoice("x", "opção"),
        "pt-BR",
    )
    assert rendered.status == "supported"
    assert rendered.premise == frame_state(TextState("assunto", "corpo"))
    assert list(rendered.input_ids[1 : rendered.pair_positions[1]]) == tokenizer.encode(
        rendered.premise
    )


def test_laya_equal_share_rejects_seventy_seven_marker_capacity() -> None:
    rendered = render_laya(
        FakeTokenizer(),
        TextState("subject", "body"),
        "instruction",
        [TextChoice(f"choice-{index}", "description") for index in range(77)],
    )
    assert rendered.status == "unsupported_capacity"


def test_laya_state_budget_includes_final_separator() -> None:
    rendered = render_laya(
        FakeTokenizer(),
        TextState("subject", "b" * 2000),
        "instruction",
        [TextChoice("choice", "description")],
    )
    assert rendered.status == "supported_with_state_truncation"
    assert len(rendered.input_ids) == 1024
    assert rendered.input_ids[-1] == 102
