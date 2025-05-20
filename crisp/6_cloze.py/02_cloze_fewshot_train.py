# 1_baseline_prompt/01_cloze_fewshot_train.py
# %%
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

random.seed(SEED)
rng = np.random.default_rng(SEED)

print(f"Running few-shot for CLOZE on {CONCEPT} with {MODEL} in train set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

FEWSHOT_EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cloze_zero_results_dev.xlsx"
)

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_cloze_few_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cloze_few_results_train.xlsx"
)

# ------------------ LOAD TRAINING DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
df = df[df.text.notna() & df.human_code.notna()]

df_fewshot_pool = df[df.train_use == "example"].copy()
df_eval = df[df.train_use == "eval"].copy()

if SAMPLE:
    df_eval = df_eval.sample(5, random_state=SEED)

# ------------------ LOAD TOP AND BOTTOM PROMPTS ------------------
results_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_prompt = results_df.loc[results_df["F1"].idxmax(), "prompt"]
bottom_prompt = results_df.loc[results_df["F1"].idxmin(), "prompt"]
top_prompt = top_prompt.replace("Text:", "")
bottom_prompt = bottom_prompt.replace("Text:", "")

# ------------------ LOAD FEWSHOT EXAMPLES ------------------
with open(FEWSHOT_EXAMPLES_PATH, "r") as f:
    sample_examples = json.load(f)
    if SAMPLE:
        sample_examples = random.sample(sample_examples, 5)


# ------------------ FORMAT PROMPTS FOR CLOZE FEWSHOT ------------------
def format_cloze_fewshot_prompt(base_prompt, examples):
    """
    Format few-shot examples as cloze completions.
    """
    example_block = "\n\n".join(
        [
            f'Text: "{ex["text"]}"\nAnswer: {"a lot" if ex["label"] == 1 else "only sometimes"}'
            for ex in examples
        ]
    )
    return f"{base_prompt.strip()}\nHere are some examples:\n{example_block}\n\n"


# ------------------ EVALUATE ------------------
response_rows = []

for sample in tqdm(sample_examples, desc="Evaluating Few-shot Cloze Prompts"):
    sample_id = sample["sample_id"]
    num_examples = sample["num_examples"]
    examples = sample["examples"]

    for category, base_prompt in [("top", top_prompt), ("bottom", bottom_prompt)]:
        prompt_text = format_cloze_fewshot_prompt(base_prompt, examples)
        prompt_id = f"{category}_fewshot_{sample_id}_n{num_examples}"

        eval_rows = classify.evaluate_prompt(
            prompt_text=prompt_text,
            prompt_id=prompt_id,
            df=df_eval,
            platform=PLATFORM,
            temperature=0.0001,
        )

        for row in eval_rows:
            row["category"] = category
            row["num_examples"] = num_examples
            row["prompt"] = prompt_text
        response_rows.extend(eval_rows)

# ------------------ EXPORT ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)

classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "category", "num_examples"],
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)
