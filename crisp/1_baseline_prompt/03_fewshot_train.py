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

# ------------------ CONSTANTS ------------------
IMPORT_RESULTS_PATH = start.MAIN_DIR + "results/ncb_variants_results_dev.xlsx"
IMPORT_PROMPT_PATH = start.DATA_DIR + "prompts/ncb_baseline_variants.xlsx"

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR + "responses_dev/ncb_variants_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = start.MAIN_DIR + "results/ncb_fewshot_results_train.xlsx"

OPENAI_API_KEY = secrets.OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

SAMPLE = True
# ------------------ LOAD DATA ------------------
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
if SAMPLE:
    df = df.sample(5)
# ------------------ LOAD PROMPTS ------------------
baseline_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
tbp_id = baseline_results.loc[baseline_results["F1"].idxmax(), "Baseline Prompt ID"]
bbp_id = baseline_results.loc[baseline_results["F1"].idxmin(), "Baseline Prompt ID"]

top_prompt = baseline_results.loc[baseline_results["F1"].idxmax(), "Prompt"]
top_prompt = top_prompt.replace("Text:", "")
bottom_prompt = baseline_results.loc[baseline_results["F1"].idxmin(), "Prompt"]
bottom_prompt = bottom_prompt.replace("Text:", "")

prompt_df = pd.read_excel(IMPORT_PROMPT_PATH, sheet_name="variants")
prompt_df = prompt_df[prompt_df.baseline_prompt_id.isin([tbp_id, bbp_id])]
# %%
# ------------------ SAMPLE EXAMPLES ------------------
df_train = df.sample(50, random_state=SEED)
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
# Add if not already:
import numpy as np


def format_fewshot_prompt(base_instruction, examples, test_text):
    fewshot_str = ""
    for ex in examples:
        fewshot_str += (
            f"Text: {ex['text']}\nAnswer: {'Yes' if ex['label'] else 'No'}\n\n"
        )
    return f"{base_instruction.strip()}\n\n{fewshot_str}Text: {test_text}"
