# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from crisp.library import start

# ------------------ SETUP ------------------
PLATFORM = start.PLATFORM
MODEL = start.MODEL

CONCEPTS = ["gratitude", "ncb", "mm"]
FIGURE_PATH = start.MAIN_DIR + f"results/{PLATFORM}_baseline_zero_f1_comparison.png"

# ------------------ LOAD DATA ------------------
dfs = {}
for concept in CONCEPTS:
    path = (
        start.MAIN_DIR
        + f"results/{PLATFORM}_{concept}_baseline_zero_results_train.xlsx"
    )
    dfs[concept] = pd.read_excel(path, sheet_name="results")

# ------------------ PLOT ------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, concept in zip(axes, CONCEPTS):
    f1_scores = dfs[concept]["F1"]
    ax.hist(
        f1_scores,
        bins=12,
        edgecolor="black",
        color="gray",
        linewidth=1.0,
    )
    ax.set_title(concept.lower(), fontsize=12)
    ax.set_xlabel("F1 Score", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="both", labelsize=9)

axes[0].set_ylabel("Frequency", fontsize=10)
plt.suptitle("F1 Score Distributions Across Baseline Prompts (Zero-Shot)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.rcParams["font.family"] = "times new roman"

plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.show()

# %%
# ------------------ CALCULATE F1 RANGES ------------------
f1_ranges = {}
for concept in CONCEPTS:
    f1_scores = dfs[concept]["F1"]
    f1_min = f1_scores.min()
    f1_max = f1_scores.max()
    f1_range = f1_max - f1_min
    f1_ranges[concept] = {
        "min": round(f1_min, 3),
        "max": round(f1_max, 3),
        "range": round(f1_range, 3),
    }

# ------------------ DISPLAY ------------------
for concept, stats in f1_ranges.items():
    print(
        f"{concept.capitalize()}: Min F1 = {stats['min']}, Max F1 = {stats['max']}, "
        f"Range = {stats['range']}"
    )
# %%
