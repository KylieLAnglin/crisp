import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from crisp.library import start

# ------------------ CONFIG ------------------
TECHNIQUE = "baseline"

# Ordered x-axis groupings
ordered_pairs = [
    ("gratitude", "openai"),
    ("ncb", "openai"),
    ("mm", "openai"),
    ("gratitude", "llama3.3"),
    ("ncb", "llama3.3"),
    ("mm", "llama3.3"),
]
group_labels = [f"{c}\n{p}" for c, p in ordered_pairs]
x_pos = np.arange(len(group_labels))
group_to_x = dict(zip(group_labels, x_pos))

# ------------------ LOAD DATA ------------------
df = pd.read_excel(os.path.join(start.RESULTS_DIR, "long_results_dev.xlsx"))
df = df[df["technique"] == TECHNIQUE]

# Pivot to wide format: zero vs few
df_wide = df.pivot_table(
    index=["platform", "concept", "category"],
    columns="few",
    values=["f1", "f1_se"],
).reset_index()
df_wide.columns = [
    "_".join([str(col) for col in c if col != ""]) for c in df_wide.columns.values
]

# Rename for clarity
df_wide = df_wide.rename(columns={"f1_False": "f1_zero", "f1_True": "f1_few"})

# Add metadata
df_wide["color"] = df_wide["category"].map({"top": "black", "bottom": "gray"})
df_wide["offset"] = df_wide["category"].map({"top": 0.1, "bottom": -0.1})
df_wide["group_label"] = df_wide["concept"] + "\n" + df_wide["platform"]
df_wide = df_wide[df_wide["group_label"].isin(group_labels)]
df_wide = df_wide.sort_values(by=["group_label", "category"])

# ------------------ PLOT ------------------
fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df_wide.iterrows():
    label = row["group_label"]
    if pd.isna(row["f1_zero"]) or pd.isna(row["f1_few"]):
        continue
    if label not in group_to_x:
        print(f"Skipping: {label} not in group_to_x")
        continue

    x = group_to_x[label] + row["offset"]
    zero = row["f1_zero"]
    few = row["f1_few"]
    color = row["color"]

    ax.plot([x, x], [zero, few], color=color, linewidth=1.0)
    ax.annotate(
        "",
        xy=(x, few),
        xytext=(x, zero),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )
    ax.scatter(x, zero, facecolors="white", edgecolors=color, s=60, zorder=3)
    ax.scatter(x, few, facecolors=color, edgecolors=color, s=60, zorder=3)

# ------------------ LEGEND ------------------
legend_elements = [
    Line2D([0], [0], color="gray", lw=1.2, label="Bottom prompt"),
    Line2D([0], [0], color="black", lw=1.2, label="Top prompt"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="white",
        label="Zero-Shot",
        markersize=7,
        linestyle="None",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        label="Few-Shot",
        markersize=7,
        linestyle="None",
    ),
]
ax.legend(handles=legend_elements, fontsize=10, frameon=False, loc="lower right")

# ------------------ STYLING ------------------
ax.set_title(
    "Few-Shot vs Zero-Shot F1 Score Comparison (Baseline, Dev Set)", fontsize=14
)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=10)
ax.set_ylim(0.3, 1)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
plt.rcParams["font.family"] = "times new roman"

plt.tight_layout()
output_path = os.path.join(start.RESULTS_DIR, "baseline_few_vs_zero_arrowplot_dev.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot: {output_path}")
