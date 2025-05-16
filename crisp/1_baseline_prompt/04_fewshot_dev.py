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
SAMPLE = start.SAMPLE
SEED = start.SEED

print(f"Running few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
BASELINE_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
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

# ------------------ EVALUATE ------------------
response_rows = []
for sample in tqdm(fewshot_samples, desc="Evaluating Few-shot Configurations"):
    fewshot_examples = sample["examples"]
    num_examples = sample["num_examples"]
    sample_id = sample["sample_id"]

    # Format few-shot examples
    intro = "Here are some examples:\n"
    examples_text = "\n".join(
        [
            f'Text: "{ex["text"]}"\nAnswer: {"Yes" if ex["label"] == 1 else "No"}'
            for ex in fewshot_examples
        ]
    )

    for prompt_label, prompt_text in [("top", top_prompt), ("bottom", bottom_prompt)]:
        full_prompt = intro + examples_text + f"\n\n{prompt_text.strip()} Text:"
        prompt_id = f"{prompt_label}_fewshot_{sample_id}_n{num_examples}"

        eval_rows = classify.evaluate_prompt(
            prompt_text=full_prompt,
            prompt_id=prompt_id,
            df=df,
            platform=PLATFORM,
            temperature=0.0001,
        )

        response_rows.extend(eval_rows)

# ------------------ EXPORT RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(RESPONSE_PATH, index=False)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)
# %%
