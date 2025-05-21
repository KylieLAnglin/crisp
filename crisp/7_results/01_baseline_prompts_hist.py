# 02_baseline_dev.py
# %%
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

# ------------------ PATHS ------------------
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)

EXPORT_FIGURE_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.png"
)
# ------------------ LOAD DATA ------------------
df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

# ------------------ PLOT HISTOGRAM ------------------
plt.figure(figsize=(8, 6))
plt.hist(
    df["F1"],
    bins=12,
    edgecolor="black",
    color="gray",  # Black & white fill
    linewidth=1.2,
)

# Style for publication
plt.title(
    "Distribution of F1 Scores for Negative Core Beliefs Prompt Variants in Training Dataset",
    fontsize=14,
)
plt.xlabel("F1 Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

# Optional: use serif fonts if desired
plt.rcParams["font.family"] = "times new roman"

plt.tight_layout()

# Show or save
plt.savefig(EXPORT_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.show()

# %%
# 02_baseline_dev.py (continued)
# %%
# 02_baseline_dev.py (continued)
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

# ------------------ PATHS ------------------
FEWSHOT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
)
FEWSHOT_EXPORT_FIGURE_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.png"
)
ZERO_DEV_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df_fewshot = pd.read_excel(FEWSHOT_RESULTS_PATH, sheet_name="results")
df_zero_dev = pd.read_excel(ZERO_DEV_RESULTS_PATH, sheet_name="results")

# ------------------ EXTRACT BASELINE F1 ------------------
top_baseline_f1 = df_zero_dev["F1"].max()
bottom_baseline_f1 = df_zero_dev["F1"].min()

# ------------------ SPLIT F1 BY CATEGORY ------------------
top_f1 = df_fewshot[df_fewshot["category"] == "top"]["F1"]
bottom_f1 = df_fewshot[df_fewshot["category"] == "bottom"]["F1"]

# ------------------ COMPUTE SHARED BINS ------------------
all_f1 = pd.concat([top_f1, bottom_f1])
bins = np.histogram_bin_edges(all_f1, bins=12)

# ------------------ PLOT OVERLAPPING HISTOGRAMS ------------------
plt.figure(figsize=(8, 6))

plt.hist(
    top_f1,
    bins=bins,
    alpha=0.6,
    label="Top Prompt (Few-Shot)",
    edgecolor="black",
    color="gray",
    linewidth=1.0,
)
plt.hist(
    bottom_f1,
    bins=bins,
    alpha=0.6,
    label="Bottom Prompt (Few-Shot)",
    edgecolor="black",
    color="white",
    hatch="//",
    linewidth=1.0,
)

# ------------------ ADD BASELINE LINES ------------------
plt.axvline(
    top_baseline_f1,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Top Prompt (Zero-Shot Baseline)",
)
plt.axvline(
    bottom_baseline_f1,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    label="Bottom Prompt (Zero-Shot Baseline)",
)

# ------------------ STYLING ------------------
plt.title("Few-Shot F1 Score Distribution by Prompt Type", fontsize=14)
plt.xlabel("F1 Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
plt.legend(fontsize=9, frameon=False)
plt.rcParams["font.family"] = "serif"

plt.tight_layout()
plt.savefig(FEWSHOT_EXPORT_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.show()

# %%
