from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[BN-PDGS {stamp}] {message}", flush=True)


__all__ = ["log"]

