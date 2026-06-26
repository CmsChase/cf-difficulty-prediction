from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLE_PATH = Path("outputs/robustness/tables/robustness_metrics_test.csv")
OUT_DIR = Path("outputs/robustness/figures")
PAPER_FIG_DIR = Path("paper/figures")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TABLE_PATH)

# Use test split only if this file accidentally contains more than test rows.
if "split_name" in df.columns:
    df = df[df["split_name"] == "test"].copy()

# Main paper robustness discussion uses HGB because Ridge is unstable in age-normalized forward-time.
df = df[df["model_name"] == "hist_gradient_boosting_regressor"].copy()

STRATEGIES = ["contest_grouped", "forward_time"]
STRATEGY_LABELS = {
    "contest_grouped": "Contest-grouped",
    "forward_time": "Forward-time",
}

COLD_ORDER = [
    "metadata_only_cold_start",
    "index_tags_only",
    "index_points_only",
    "tags_points_only",
    "full_api_reference",
]
COLD_LABELS = {
    "metadata_only_cold_start": "metadata\nonly",
    "index_tags_only": "index\n+ tags",
    "index_points_only": "index\n+ points",
    "tags_points_only": "tags\n+ points",
    "full_api_reference": "full API\nreference",
}

AGE_ORDER = [
    "raw_solved_only_reference",
    "age_normalized_solved_only",
    "index_tags_points_age_norm",
    "full_api_without_raw_solved_but_with_age_norm",
    "full_api_plus_age_norm",
]
AGE_LABELS = {
    "raw_solved_only_reference": "raw solved\nonly",
    "age_normalized_solved_only": "age-normalized\nsolved only",
    "index_tags_points_age_norm": "metadata\n+ age norm",
    "full_api_without_raw_solved_but_with_age_norm": "full API no raw\n+ age norm",
    "full_api_plus_age_norm": "full API\n+ age norm",
}

def plot_two_panel(data, feature_order, label_map, metric, ylabel, title, filename, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, strategy in zip(axes, STRATEGIES):
        sub = data[data["strategy"] == strategy].copy()
        vals = []
        labels = []

        for feat in feature_order:
            row = sub[sub["feature_set_name"] == feat]
            if len(row) == 0:
                continue
            vals.append(float(row.iloc[0][metric]))
            labels.append(label_map.get(feat, feat))

        x = np.arange(len(vals))
        ax.bar(x, vals, width=0.65)

        ax.set_title(STRATEGY_LABELS[strategy])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.grid(axis="y", alpha=0.3)

        if ylim is not None:
            ax.set_ylim(*ylim)

        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = OUT_DIR / filename
    paper_path = PAPER_FIG_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(paper_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

plot_two_panel(
    data=df[df["feature_set_name"].isin(COLD_ORDER)],
    feature_order=COLD_ORDER,
    label_map=COLD_LABELS,
    metric="MAE",
    ylabel="MAE",
    title="Cold-start robustness: test MAE",
    filename="cold_start_mae_comparison.png",
    ylim=(0, 390),
)

plot_two_panel(
    data=df[df["feature_set_name"].isin(COLD_ORDER)],
    feature_order=COLD_ORDER,
    label_map=COLD_LABELS,
    metric="within_200",
    ylabel="Within 200 rating points",
    title="Cold-start robustness: within 200 rating points",
    filename="cold_start_within_200_comparison.png",
    ylim=(0, 0.8),
)

plot_two_panel(
    data=df[df["feature_set_name"].isin(AGE_ORDER)],
    feature_order=AGE_ORDER,
    label_map=AGE_LABELS,
    metric="MAE",
    ylabel="MAE",
    title="Age-normalized solved-count robustness: test MAE",
    filename="age_normalized_mae_comparison.png",
    ylim=(0, 460),
)

print("Redrew robustness figures:")
print(OUT_DIR / "cold_start_mae_comparison.png")
print(OUT_DIR / "cold_start_within_200_comparison.png")
print(OUT_DIR / "age_normalized_mae_comparison.png")
print("Also copied them to paper/figures/")
