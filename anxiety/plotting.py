"""Small helpers so every script saves figures the same way."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save to disk, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402

sns.set_theme(style="whitegrid", palette="viridis")


def save(fig_name: str, subdir: str = "") -> Path:
    out_dir = config.FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
