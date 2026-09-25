from __future__ import annotations

from typing import Any

from .biot_adapter import RepositoryExtractorAdapter


class CBraModAdapter(RepositoryExtractorAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("cbramod", config)


class LaBraMAdapter(RepositoryExtractorAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("labram", config)


__all__ = ["CBraModAdapter", "LaBraMAdapter"]
