"""Generate the project overview diagram used by the README.

The diagram is intentionally simple and static: it documents the completed v5
research pipeline without depending on generated data or experiment outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "project_overview.png"


def _add_box(
    ax: plt.Axes,
    *,
    center_x: float,
    center_y: float,
    text: str,
    width: float,
    height: float,
    facecolor: str,
) -> None:
    """Add a rounded labeled box to an axes."""

    x0 = center_x - width / 2
    y0 = center_y - height / 2
    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.4,
        edgecolor="#334155",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        center_x,
        center_y,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#0f172a",
        linespacing=1.25,
    )


def _add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Add a clean directional arrow between pipeline stages."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color="#475569",
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(arrow)


def main() -> None:
    """Render ``docs/project_overview.png``."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = [
        (1.4, 4.25, "Codeforces API\n+ problem pages", "#dbeafe"),
        (3.5, 4.25, "Raw JSON snapshots\n+ cached HTML", "#e0f2fe"),
        (5.6, 4.25, "Preprocessing\n+ feature engineering", "#dcfce7"),
        (7.7, 4.25, "Metadata\nsolved statistics\nstatement text-light", "#fef9c3"),
        (9.8, 4.25, "Evaluation splits\ncontest-grouped\nforward-time", "#fae8ff"),
        (11.9, 4.25, "Baseline models\nablations\nrobustness checks", "#ffedd5"),
        (11.9, 1.75, "Analysis artifacts\npaper\nREADME\nreviewer guide", "#f1f5f9"),
    ]

    for center_x, center_y, text, facecolor in nodes:
        _add_box(
            ax,
            center_x=center_x,
            center_y=center_y,
            text=text,
            width=1.75,
            height=1.05,
            facecolor=facecolor,
        )

    for left, right in zip(nodes[:5], nodes[1:6]):
        _add_arrow(ax, (left[0] + 0.95, left[1]), (right[0] - 0.95, right[1]))

    _add_arrow(ax, (11.9, 3.65), (11.9, 2.35))

    ax.text(
        7,
        5.45,
        "cf-difficulty-prediction v5 research pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        7,
        0.6,
        "Official API data and cached problem pages feed reproducible tabular experiments; generated data and outputs stay out of git.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
