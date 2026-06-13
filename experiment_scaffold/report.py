"""Aggregate training outputs (Lightning-style metrics.csv) into DIGEST rows."""
from pathlib import Path

import pandas as pd


def read_csv_metrics(metrics_csv) -> dict[str, float]:
    """Last non-null value of each metric column in a Lightning metrics.csv."""
    df = pd.read_csv(metrics_csv)
    out: dict[str, float] = {}
    for col in df.columns:
        if col in ("epoch", "step"):
            continue
        s = df[col].dropna()
        if len(s):
            out[col] = float(s.iloc[-1])
    return out


def digest_row(exp_id, sweep, hypothesis, metrics: dict[str, float],
               conclusion) -> str:
    """Build one append-only DIGEST.md table row."""
    keymetrics = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    return f"| {exp_id} | {sweep} | {hypothesis} | {keymetrics} | {conclusion} |"


def append_digest(digest_path, row: str, header: str | None = None) -> None:
    """Append ``row`` to DIGEST.md, writing ``header`` first if file is new."""
    p = Path(digest_path)
    if not p.exists() and header:
        p.write_text(header.rstrip() + "\n")
    with p.open("a") as f:
        f.write(row.rstrip() + "\n")
