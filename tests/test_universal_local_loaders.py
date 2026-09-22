from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType
from typing import Any, cast

import pytest

from benchmarks.universal_local import loaders
from benchmarks.universal_local.registry import Candidate


class FakeTensor:
    def __init__(self, shape: list[int], dtype: object, values: object = None) -> None:
        self.shape = shape
        self.dtype = dtype
        self._values = values
        self.copied = False

    def float(self) -> FakeTensor:
        return self

    def copy_(self, value: FakeTensor) -> None:
        self.copied = value is self or isinstance(value, FakeTensor)

    def tolist(self) -> object:
        return self._values


class FakeSlice:
    def __init__(self, tensor: FakeTensor, dtype: str) -> None:
        self.tensor = tensor
        self.dtype = dtype

    def get_shape(self) -> list[int]:
        return self.tensor.shape

    def get_dtype(self) -> str:
        return self.dtype


class FakeSource:
    def __init__(self, tensors: dict[str, tuple[FakeTensor, str]]) -> None:
        self.tensors = tensors

    def __enter__(self) -> FakeSource:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def keys(self) -> list[str]:
        return list(self.tensors)

    def get_slice(self, key: str) -> FakeSlice:
        tensor, dtype = self.tensors[key]
        return FakeSlice(tensor, dtype)

    def get_tensor(self, key: str) -> FakeTensor:
        return self.tensors[key][0]


class FakeReceipt:
    def assert_live(self) -> None:
        return None

    def descriptor_path(self, path: str) -> str:
        assert path == "model.safetensors"
        return "/dev/fd/fake"


class FakeTorch:
    int64 = "i64"

    @staticmethod
    def no_grad() -> Any:
        return nullcontext()


class FakeModel:
    def __init__(self, state: dict[str, FakeTensor]) -> None:
        self.state = state

    def state_dict(self) -> dict[str, FakeTensor]:
        return self.state


def _candidate(identifier: str, dtypes: dict[str, int]) -> Candidate:
    return Candidate({"id": identifier, "source_weight_dtypes": dtypes})


def _install_source(monkeypatch: pytest.MonkeyPatch, source: FakeSource) -> None:
    module = ModuleType("safetensors")
    module.safe_open = lambda *_args, **_kwargs: source  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", module)


def test_strict_loader_keeps_only_laya_temperature_as_f32(monkeypatch: pytest.MonkeyPatch) -> None:
    weight = FakeTensor([2, 2], "f32")
    temperature = FakeTensor([3], "f32")
    _install_source(
        monkeypatch, FakeSource({"weight": (weight, "F16"), "temperature": (temperature, "F32")})
    )
    loaders._load_strict(
        cast(Any, FakeReceipt()),
        _candidate("laya-multilingual", {"F16": 1, "F32": 1}),
        FakeModel({"weight": weight, "temperature": temperature}),
        FakeTorch(),
    )
    assert weight.copied and temperature.copied


def test_strict_loader_removes_only_exact_mdeberta_position_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = FakeTensor([2, 2], "f32")
    position = FakeTensor([1, 512], "i64", [list(range(512))])
    _install_source(
        monkeypatch,
        FakeSource(
            {"weight": (weight, "F16"), "deberta.embeddings.position_ids": (position, "I64")}
        ),
    )
    loaders._load_strict(
        cast(Any, FakeReceipt()),
        _candidate("mdeberta-nli", {"F16": 1, "I64": 1}),
        FakeModel({"weight": weight}),
        FakeTorch(),
    )
    assert weight.copied


def test_strict_loader_rejects_unknown_tensor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    weight = FakeTensor([2, 2], "f32")
    _install_source(
        monkeypatch,
        FakeSource({"unexpected": (weight, "F16"), "temperature": (FakeTensor([3], "f32"), "F32")}),
    )
    with pytest.raises(loaders.LoadError, match="key mismatch"):
        loaders._load_strict(
            cast(Any, FakeReceipt()),
            _candidate("laya-multilingual", {"F16": 1, "F32": 1}),
            FakeModel({"weight": weight, "temperature": FakeTensor([3], "f32")}),
            FakeTorch(),
        )
