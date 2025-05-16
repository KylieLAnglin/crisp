# 1_baseline_prompt/04_fewshot_dev.py
# %%
import os
import json
import pandas as pd
from tqdm import tqdm

from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = True
SEED = start.SEED

print(f"Running few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
BASELINE_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_fewshot_responses_dev.xlsx"
)
RESULTS_PATH = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_fewshot_results_dev.xlsx"

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD FEWSHOT EXAMPLES ------------------
with open(EXAMPLES_PATH, "r") as f:
    fewshot_samples = json.load(f)

# ------------------ LOAD TOP & BOTTOM BASELINE PROMPTS ------------------
baseline_df = pd.read_excel(BASELINE_RESULTS_PATH, sheet_name="results")
top_prompt = baseline_df.loc[baseline_df["F1"].idxmax(), "prompt"]
bottom_prompt = baseline_df.loc[baseline_df["F1"].idxmin(), "prompt"]

# ------------------ LOAD TRAINING FEWSHOT RESULTS ------------------
train_fewshot_results_path = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_fewshot_results_train.xlsx"
)
train_df = pd.read_excel(train_fewshot_results_path, sheet_name="results")

# ------------------ IDENTIFY TOP TRAINING SAMPLE PER CATEGORY ------------------
best_samples = {}
for category in ["top", "bottom"]:
    category_df = train_df[train_df["category"] == category]
    best_row = category_df.loc[category_df["F1"].idxmax()]
    best_samples[category] = {
        "prompt_id": best_row["prompt_id"],
        "sample_id": int(
            best_row["prompt_id"].split("_")[2]
        ),  # expects format: top_fewshot_12_nX
    }

print(f"Selected best sample IDs: {best_samples}")

# ------------------ FETCH BEST EXAMPLES ------------------
selected_examples = {}
for category, sample_info in best_samples.items():
    sample_id = sample_info["sample_id"]
    matched = [s for s in fewshot_samples if s["sample_id"] == sample_id]
    assert len(matched) == 1, f"Expected one match for sample_id={sample_id}"
    selected_examples[category] = matched[0]["examples"]

# ------------------ EVALUATE ON DEV SET ------------------
response_rows = []
for category, base_prompt in [("top", top_prompt), ("bottom", bottom_prompt)]:
    example_block = "\n".join(
        [
            f'Text: "{ex["text"]}"\nAnswer: {"Yes" if ex["label"] == 1 else "No"}'
            for ex in selected_examples[category]
        ]
    )
    base_prompt = base_prompt.replace("Text:", "")
    full_prompt = f"{base_prompt}\nHere are some examples:\n{example_block}\n\n"
    prompt_id = f"{category}_fewshot_best_n{len(selected_examples[category])}"

    eval_rows = classify.evaluate_prompt(
        prompt_text=full_prompt,
        prompt_id=prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=0.0001,
    )

    for row in eval_rows:
        row["category"] = category
        row["num_examples"] = len(selected_examples[category])
    response_rows.extend(eval_rows)

# ------------------ EXPORT ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(RESPONSE_PATH, index=False)

classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col=["prompt_id", "category", "num_examples"],
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)
