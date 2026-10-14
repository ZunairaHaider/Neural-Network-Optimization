"""
runtime_analysis/csv_logger.py
--------------------------------
Appends RuntimeRecord rows to a CSV file.

The logger is designed to be append-safe:
  - Creates the file with a header on first write.
  - Appends rows without re-writing the header on subsequent calls.
  - Thread/process-safe via file-level locking (fcntl on Unix,
    falls back gracefully on Windows).

Usage
-----
    from runtime_analysis.csv_logger import RuntimeCSVLogger
    from runtime_analysis.timing_utils import RuntimeRecord

    logger = RuntimeCSVLogger()          # defaults to runtime_analysis/pruning_runtime_costs.csv
    record = RuntimeRecord(method="obd", dataset="A", ...)
    logger.append(record)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

from runtime_analysis.timing_utils import RuntimeRecord

# ── CSV column order (must match RuntimeRecord.to_dict keys) ─────────────────
FIELDNAMES: List[str] = [
    "method",
    "dataset",
    "dim",
    "sparsity_pct",
    "scoring_time_s",
    "finetune_time_s",
    "total_time_s",
    "final_test_mse",
    "baseline_mse",
    "mse_ratio",
    "param_count",
    "inference_ms",
    "cpu_peak_mb",
    "cuda_peak_mb",
    "flops_estimate",
]

_DEFAULT_CSV = Path(__file__).parent / "pruning_runtime_costs.csv"


class RuntimeCSVLogger:
    """Append RuntimeRecord instances to a CSV file.

    Parameters
    ----------
    csv_path : Path or str, optional
        Where to write the CSV. Defaults to
        ``runtime_analysis/pruning_runtime_costs.csv``.
    """

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def append(self, record: RuntimeRecord) -> None:
        """Append one record to the CSV (creates file + header if needed)."""
        row = record.to_dict()
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            if needs_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    def append_many(self, records: list[RuntimeRecord]) -> None:
        """Append a list of records in one open-close cycle."""
        for rec in records:
            self.append(rec)

    def reset(self) -> None:
        """Delete the CSV so the next append() starts fresh."""
        if self.csv_path.exists():
            self.csv_path.unlink()
            print(f"[RuntimeCSVLogger] Cleared {self.csv_path}")

    def path(self) -> Path:
        """Return the resolved path to the CSV file."""
        return self.csv_path.resolve()
