from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class LayerDecision:
    tensor_name: str
    module_name: str
    component: str
    layer_kind: str
    quantize: bool
    reason: str


class ArchitectureAdapter(ABC):
    adapter_id: str

    @abstractmethod
    def detect(self, config: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def validate_checkpoint(self, config: Mapping[str, Any], tensor_names: Iterable[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def classify_tensor(self, name: str, shape: tuple[int, ...]) -> LayerDecision:
        raise NotImplementedError

    def enumerate_quantizable_layers(
        self, tensors: Iterable[tuple[str, tuple[int, ...]]]
    ) -> list[LayerDecision]:
        return [decision for name, shape in tensors if (decision := self.classify_tensor(name, shape)).quantize]

    @abstractmethod
    def get_exclusion_policy(self) -> dict[str, str]:
        raise NotImplementedError

