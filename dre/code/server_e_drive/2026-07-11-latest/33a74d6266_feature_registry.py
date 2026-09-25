from __future__ import annotations

from dataclasses import dataclass


class FeatureRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class RegisteredFeature:
    name: str
    group: str
    source: str
    uses_label: bool


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: list[RegisteredFeature] = []

    def add(self, name: str, *, group: str, source: str, uses_label: bool) -> None:
        lower = name.lower()
        if uses_label or "oracle" in lower or "clinical" in lower or lower in {"center", "subject_id", "patient_id"}:
            raise FeatureRegistryError(f"forbidden A12 model feature: {name}")
        if any(item.name == name for item in self._features):
            raise FeatureRegistryError(f"duplicate feature: {name}")
        self._features.append(RegisteredFeature(name, group, source, uses_label))

    def names(self, groups: set[str] | None = None) -> list[str]:
        return [item.name for item in self._features if groups is None or item.group in groups]

    def to_dict(self) -> dict[str, object]:
        return {"features": [item.__dict__ for item in self._features]}
