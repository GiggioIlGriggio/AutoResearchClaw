"""Reusable research plotting + tables: house style, CI curves, FC heatmaps,
markdown/CSV tables. Uses the Agg backend (headless-safe) and a colorblind-safe
Okabe-Ito palette.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402

OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]


def apply_style() -> None:
    """Apply the shared research plotting style in-place."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "image.cmap": "viridis",
        "font.size": 11,
    })


def curve_with_ci(x, ys, labels, xlabel, ylabel, title, out_path,
                  logx: bool = True) -> Path:
    """Plot mean +/- std curves. ``ys``: list of (n_seeds, n_points) arrays."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for y, lab in zip(ys, labels):
        y = np.asarray(y)
        ax.errorbar(x, y.mean(0), yerr=y.std(0), marker="o", capsize=3,
                    label=lab)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fc_matrix(fc, out_path, vmax: float | None = None) -> Path:
    """Heatmap of a functional-connectivity matrix (diverging cmap, 0-centered)."""
    apply_style()
    fc = np.asarray(fc)
    if vmax is None:
        vmax = float(np.abs(fc).max())
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_table(df: pd.DataFrame, path_stem) -> tuple[Path, Path]:
    """Write ``df`` to ``<stem>.md`` and ``<stem>.csv``.

    Returns
    -------
    tuple[Path, Path]
        (markdown_path, csv_path).
    """
    stem = Path(path_stem)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(df) + "\n")
    return md_path, csv_path


def _to_markdown(df: pd.DataFrame) -> str:
    """DataFrame -> GitHub markdown table. Uses pandas' ``to_markdown`` when the
    optional ``tabulate`` package is present, else a dependency-free fallback."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in df.columns]
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in df.itertuples(index=False, name=None)]
        return "\n".join([head, sep, *rows])
