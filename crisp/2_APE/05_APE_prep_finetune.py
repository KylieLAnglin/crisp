import pandas as pd
import json
from tqdm import tqdm
from crisp.library import start

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
SEED = start.SEED
PLATFORM = start.PLATFORM

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_APE_zero_results_dev.xlsx"
)
EXPORT_DIR = start.DATA_DIR + f"finetune/"

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]

# ------------------ LOAD PROMPTS ------------------
train_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

prompt_df = train_results[train_results["prompt_id"].isin([top_id, bottom_id])]
prompt_map = {
    "top": train_results.loc[train_results["prompt_id"] == top_id, "prompt"].values[0],
    "bottom": train_results.loc[
        train_results["prompt_id"] == bottom_id, "prompt"
    ].values[0],
}

# ------------------ EXPORT JSONL FILES ------------------
for label, prompt_text in prompt_map.items():
    prompt_cleaned = prompt_text.replace("Text:", "").strip()
    jsonl_path = f"{EXPORT_DIR}{PLATFORM}_{CONCEPT}_APE_finetune_train_{label}.jsonl"

    rows = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Building data for {label} prompt"
    ):
        user_input = f"{prompt_text.strip()} {row['text'].strip()}"
        expected_output = "Yes" if row["human_code"] == 1 else "No"
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": prompt_cleaned},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": expected_output},
                ]
            }
        )

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(rows)} examples to {jsonl_path}")
