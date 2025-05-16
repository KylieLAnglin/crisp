# 1_baseline_prompt/03_fewshot_train.py
# %%
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

print(f"Running few-shot setup and evaluation on {CONCEPT} with {MODEL}")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)
FEWSHOT_EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_fewshot_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_fewshot_results_train.xlsx"
)

# ------------------ LOAD TRAINING DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
df = df[df.text.notna() & df.human_code.notna()]

# Split data: 50 for few-shot pool, remainder for evaluation
df_fewshot_pool = df.sample(50, random_state=SEED)
df_eval = df.drop(df_fewshot_pool.index)
if SAMPLE:
    df_eval = df_eval.sample(5, random_state=SEED)
# ------------------ LOAD TOP AND BOTTOM PROMPTS ------------------
results_df = pd.read_excel(RESULTS_PATH, sheet_name="results")
top_prompt = results_df.loc[results_df["F1"].idxmax(), "prompt"]
bottom_prompt = results_df.loc[results_df["F1"].idxmin(), "prompt"]
top_prompt = top_prompt.replace("Text:", "")
bottom_prompt = bottom_prompt.replace("Text:", "")

# ------------------ GENERATE FEWSHOT SAMPLES ------------------
sample_examples = []
for sample_num in range(1, 51):
    num_examples = np.random.randint(1, 11)
    sampled = df_fewshot_pool.sample(num_examples, random_state=sample_num)
    examples = [
        {"text": text, "label": label}
        for text, label in zip(sampled["text"], sampled["human_code"])
    ]
    sample_examples.append(
        {"sample_id": sample_num, "num_examples": num_examples, "examples": examples}
    )

# ------------------ SAVE SAMPLE DEFINITIONS ------------------
os.makedirs(os.path.dirname(FEWSHOT_EXAMPLES_PATH), exist_ok=True)
with open(FEWSHOT_EXAMPLES_PATH, "w") as f:
    json.dump(sample_examples, f, indent=2)
print(f"Saved {len(sample_examples)} few-shot samples to {FEWSHOT_EXAMPLES_PATH}")

# ------------------ EVALUATE SAMPLES ------------------
response_rows = []
for sample in tqdm(sample_examples, desc="Evaluating Few-shot Samples"):
    sample_id = sample["sample_id"]
    num_examples = sample["num_examples"]
    example_block = "\n".join(
        [
            f'Text: "{ex["text"]}"\nAnswer: {"Yes" if ex["label"] == 1 else "No"}'
            for ex in sample["examples"]
        ]
    )

    for category, base_prompt in [("top", top_prompt), ("bottom", bottom_prompt)]:
        full_prompt = (
            f"Here are some examples:\n{example_block}\n\n{base_prompt.strip()} Text:"
        )
        prompt_id = f"{category}_fewshot_{sample_id}_n{num_examples}"

        eval_rows = classify.evaluate_prompt(
            prompt_text=full_prompt,
            prompt_id=prompt_id,
            df=df_eval,
            platform=PLATFORM,
            temperature=0.0001,
        )

        for row in eval_rows:
            row["category"] = category
            row["num_examples"] = num_examples
        response_rows.extend(eval_rows)

# ------------------ EXPORT RESPONSES AND METRICS ------------------
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
# %%
