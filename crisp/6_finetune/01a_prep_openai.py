import os
import pandas as pd
import json
from tqdm import tqdm
from crisp.library import start

# ------------------ CONFIG ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
TECHNIQUES = ["baseline", "APE", "persona"]
LABELS = ["top", "bottom"]
PLATFORM = "openai"  # assuming constant for this run

EXPORT_DIR = start.DATA_DIR + "finetune/"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ------------------ GENERATE FILES ------------------
for concept in CONCEPTS:
    for technique in TECHNIQUES:
        print(f"\n📚 Processing: {concept} — {technique}")

        # Load training split
        data_path = os.path.join(start.DATA_DIR, f"clean/{concept}.xlsx")
        df = pd.read_excel(data_path)
        df = df[df.split_group == "train"]

        # Load result file for prompt selection
        results_path = os.path.join(
            start.MAIN_DIR,
            f"results/{PLATFORM}_{concept}_{technique.lower()}_zero_results_dev.xlsx",
        )
        if not os.path.exists(results_path):
            print(f"❌ Skipping — missing file: {results_path}")
            continue

        train_results = pd.read_excel(results_path, sheet_name="results")
        top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
        bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

        prompt_map = {
            "top": train_results.loc[
                train_results["prompt_id"] == top_id, "prompt"
            ].values[0],
            "bottom": train_results.loc[
                train_results["prompt_id"] == bottom_id, "prompt"
            ].values[0],
        }

        for label in LABELS:
            prompt_text = prompt_map[label]
            prompt_cleaned = prompt_text.replace("Text:", "").strip()

            jsonl_path = os.path.join(
                EXPORT_DIR,
                f"{PLATFORM}_{concept}_{technique}_finetune_train_{label}.jsonl",
            )

            rows = []
            for _, row in tqdm(
                df.iterrows(), total=len(df), desc=f"🔨 {concept}-{technique}-{label}"
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

            print(f"✅ Saved {len(rows)} examples to {jsonl_path}")
