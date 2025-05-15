# A2_baseline_dev.py
# %%
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook, Workbook
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
from crisp.library import start, secrets
from crisp.library import format_prompts, classify, metric_standard_errors
import numpy as np

# ------------------ CONSTANTS ------------------

CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in train set")
SAMPLE = start.SAMPLE

IMPORT_PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"


IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_baseline_few_responses_dev.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
if SAMPLE:
    df = df.sample(5)
# ------------------ LOAD PROMPTS ------------------
training_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
tbp_id = training_results.loc[training_results["F1"].idxmax(), "Baseline Prompt ID"]
bbp_id = training_results.loc[training_results["F1"].idxmin(), "Baseline Prompt ID"]

prompt_df = pd.read_excel(IMPORT_PROMPT_PATH, sheet_name="baseline")
prompt_df = prompt_df[prompt_df.baseline_prompt_id.isin([tbp_id, bbp_id])]
prompt_df["prompt"] = prompt_df["prompt"].str.replace("Text: ", "")

# %%
# ------------------ SAMPLE EXAMPLES ------------------
df_train = df.sample(50, random_state=start.SEED)
df_test = df.drop(df_train.index)
df_train["label"] = np.where(df_train.human_code == 1, "Yes", "No")

sample_examples = []
for sample_num in range(1, 51):
    # random sample num_examples 1 through 10
    num_examples = np.random.randint(1, 11)

    # random sample from df_train without replacement
    train_sample = df_train.sample(num_examples, random_state=sample_num)

    texts = train_sample["text"].tolist()
    labels = train_sample["label"].tolist()

    sample_examples.append(
        {
            "num_examples": num_examples,
            "examples": [
                {"text": text, "label": label} for text, label in zip(texts, labels)
            ],
        }
    )
# %%
# ------------------ FORMAT PROMPTS ------------------


def format_fewshot_prompt(base_instruction, examples, test_text):
    fewshot_str = ""
    for ex in examples:
        fewshot_str += (
            f"Text: {ex['text']}\nAnswer: {'Yes' if ex['label'] else 'No'}\n\n"
        )
    return f"{base_instruction.strip()}\n\n{fewshot_str}Text: {test_text}"


# %%
# ------------------ COLLECT RESPONSES ------------------
