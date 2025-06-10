# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from crisp.library import start

# ------------------ CONFIG ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]
RESULTS_DIR = start.MAIN_DIR + "results/"
EXPORT_PATH = RESULTS_DIR + "min_max_f1_dev.png"

# ------------------ COLLECT MIN/MAX F1 ------------------
summary_rows = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        file_path = os.path.join(
            RESULTS_DIR, f"{platform}_{concept}_baseline_zero_results_dev.xlsx"
        )
        df = pd.read_excel(file_path, sheet_name="results")

        min_f1 = df["F1"].min()
        max_f1 = df["F1"].max()

        summary_rows.append(
            {
                "Concept": concept.lower(),
                "Model": platform,
                "Min F1": min_f1,
                "Max F1": max_f1,
            }
        )

summary_df = pd.DataFrame(summary_rows)

# ------------------ PLOT ------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Define positions
x_labels = [f"{row['Concept']}\n{row['Model']}" for _, row in summary_df.iterrows()]
x_pos = np.arange(len(x_labels))

# Plot min/max as vertical lines
for i, row in summary_df.iterrows():
    ax.plot(
        [x_pos[i], x_pos[i]],
        [row["Min F1"], row["Max F1"]],
        color="black",
        linewidth=2.0,
    )
    ax.scatter(
        x_pos[i],
        row["Min F1"],
        color="gray",
        edgecolor="black",
        s=40,
        zorder=3,
    )
    ax.scatter(
        x_pos[i],
        row["Max F1"],
        color="white",
        edgecolor="black",
        s=40,
        zorder=3,
    )

# ------------------ STYLING ------------------
ax.set_title("Range of F1 Scores by Concept and Model (Dev Dataset)", fontsize=14)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()

# Use serif font for publication style
plt.rcParams["font.family"] = "times new roman"

# ------------------ SAVE ------------------
plt.savefig(EXPORT_PATH, dpi=300, bbox_inches="tight")
plt.show()
