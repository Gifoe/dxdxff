from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def log(message: str) -> None:
    print(f"[HUP-StrictInterictal] {message}", flush=True)


def parse_csv_set(value: str) -> set[str]:
    return {part.strip().lower() for part in re.split(r"[,;\s]+", str(value)) if part.strip()}


def norm_text(value: object) -> str:
    return str(value).strip().lower()


def safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def run_cmd(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)
    log(printable)
    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fout.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
