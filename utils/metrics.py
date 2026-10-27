import csv
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.loggings import log_main


@contextmanager
def time_into_log(
    log_dict: dict[str, Any], label: str, precision: int = 2, echo: bool = False
) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    duration_ms = round((time.perf_counter() - start) * 1000, precision)
    log_dict[label] = duration_ms

    if echo:
        log_main(f"Timed block {label} completed in {duration_ms:.2f} ms")


def log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with path.open() as f:
            if f.readline().strip().split(",") != list(row):
                path.rename(path.with_suffix(f".{int(time.time())}.csv"))
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
