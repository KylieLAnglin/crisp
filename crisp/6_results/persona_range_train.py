import os
import pandas as pd
from crisp.library import start

# ------------------ CONFIG ------------------
RESULTS_DIR = start.RESULTS_DIR  # CHANGE THIS TO YOUR ACTUAL RESULTS DIR
PLATFORMS = ["openai", "llama3.3"]
CONCEPTS = ["gratitude", "ncb", "mm"]
CATEGORIES = ["bottom", "top"]
TECHNIQUE = "persona"
SETTING = "zero"
SPLIT = "train"

# ------------------ COLLECT BEST/WORST F1s ------------------
records = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        filename = f"{platform}_{concept}_{TECHNIQUE}_{SETTING}_results_{SPLIT}.xlsx"
        file_path = os.path.join(RESULTS_DIR, filename)

        try:
            df = pd.read_excel(file_path, sheet_name="results")

            # Add category if not present
            if "category" not in df.columns:
                max_f1 = df["F1"].max()
                df["category"] = df["F1"].apply(
                    lambda f: "top" if f == max_f1 else "bottom"
                )

            for category in CATEGORIES:
                subset = df[df["category"] == category]
                if subset.empty:
                    continue

                best = subset.loc[subset["F1"].idxmax()]
                worst = subset.loc[subset["F1"].idxmin()]

                records.append(
                    {
                        "platform": platform,
                        "concept": concept,
                        "category": category,
                        "best_f1": best["F1"],
                        "best_prompt_id": best["prompt_id"],
                        "worst_f1": worst["F1"],
                        "worst_prompt_id": worst["prompt_id"],
                    }
                )
        except Exception as e:
            print(f"Could not load {file_path}: {e}")

# ------------------ BUILD & SORT SUMMARY DATAFRAME ------------------
summary_df = pd.DataFrame(records)

# Categorical sort for clean ordering
summary_df["platform"] = pd.Categorical(
    summary_df["platform"], categories=PLATFORMS, ordered=True
)
summary_df["concept"] = pd.Categorical(
    summary_df["concept"], categories=CONCEPTS, ordered=True
)
summary_df["category"] = pd.Categorical(
    summary_df["category"], categories=CATEGORIES, ordered=True
)

summary_df = summary_df.sort_values(by=["platform", "concept", "category"])
summary_df = summary_df[
    [
        "platform",
        "concept",
        "category",
        "worst_f1",
        "best_f1",
    ]
]
summary_df["difference"] = summary_df["best_f1"] - summary_df["worst_f1"]
# Save if desired
summary_df.to_excel(
    os.path.join(RESULTS_DIR, "persona_prompt_extremes_sorted.xlsx"), index=False
)

# Display in console
print(summary_df)
