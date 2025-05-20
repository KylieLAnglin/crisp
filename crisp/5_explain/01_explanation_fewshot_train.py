# 1_baseline_prompt/01_explanation_fewshot_train.py
# %%
import os
import json
from itertools import product
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

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
BASELINE_PROMPT_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_explanation_few_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_train.xlsx"
)

# ------------------ LOAD TRAINING DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[(df.split_group == "train") & (df.text.notna()) & (df.human_code.notna())]
df = df[df.train_use == "eval"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD BEST PROMPTS ------------------
prompt_df = pd.read_excel(BASELINE_PROMPT_PATH, sheet_name="results")
top_prompt = (
    prompt_df.loc[prompt_df["F1"].idxmax(), "prompt"].replace("Text:", "").strip()
)
bottom_prompt = (
    prompt_df.loc[prompt_df["F1"].idxmin(), "prompt"].replace("Text:", "").strip()
)

# ------------------ LOAD FEWSHOT EXAMPLES ------------------
with open(EXAMPLES_PATH, "r") as f:
    fewshot_samples = json.load(f)


# ------------------ GENERATE EXPLANATION PROMPT VARIANTS ------------------
def generate_explanation_prompt_variants(example_set, base_prompt, id_prefix="explain"):
    """
    Construct explanation-based few-shot prompts for a given example set.
    """
    sample_id = example_set["sample_id"]
    num_examples = example_set["num_examples"]
    examples = example_set["examples"]

    parts = []
    for ex in examples:
        answer = "Yes" if ex["label"] == 1 else "No"
        block = (
            f'Text: "{ex["text"]}"\nAnswer: {answer} Explanation: {ex["explanation"]}'
        )
        parts.append(block)

    example_block = "\n\n".join(parts)
    full_prompt = f"{base_prompt}\nHere are some examples:\n{example_block}\n\n"
    prompt_id = f"{id_prefix}_{sample_id}_n{num_examples}"

    return {
        "prompt_id": prompt_id,
        "prompt_block": full_prompt,
        "num_examples": num_examples,
    }


# ------------------ EVALUATE PROMPTS ------------------
response_rows = []
random.seed(SEED)

# Select up to 50 few-shot samples
fewshot_samples = random.sample(fewshot_samples, min(50, len(fewshot_samples)))
if SAMPLE:
    fewshot_samples = fewshot_samples[:5]

for sample in tqdm(fewshot_samples, desc="Evaluating Explanation Prompts"):
    for category, base_prompt in [("top", top_prompt), ("bottom", bottom_prompt)]:
        variant = generate_explanation_prompt_variants(
            example_set=sample, base_prompt=base_prompt, id_prefix=f"{category}_explain"
        )

        eval_rows = classify.evaluate_prompt(
            prompt_text=variant["prompt_block"],
            prompt_id=variant["prompt_id"],
            df=df,
            platform=PLATFORM,
            temperature=0.0001,
        )

        for row in eval_rows:
            row["prompt_id"] = variant["prompt_id"]
            row["category"] = category
            row["num_examples"] = variant["num_examples"]
            row["prompt"] = variant["prompt_block"]

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
