# 02_baseline_dev.py
# %%
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook, Workbook
from tqdm import tqdm
from datetime import datetime
import os

from crisp.library import start, secrets
from crisp.library import format_prompts, classify, metric_standard_errors

CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in dev set")
# ------------------ CONSTANTS ------------------
IMPORT_PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_results_train.xlsx"
)

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_baseline_responses_dev.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
if start.SAMPLE:
    df = df.sample(5, random_state=start.SEED)

# ------------------ LOAD PROMPTS ------------------
training_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
tbp_id = training_results.loc[training_results["F1"].idxmax(), "Baseline Prompt ID"]
bbp_id = training_results.loc[training_results["F1"].idxmin(), "Baseline Prompt ID"]

prompt_df = pd.read_excel(IMPORT_PROMPT_PATH, sheet_name="baseline")
prompt_df = prompt_df[prompt_df.baseline_prompt_id.isin([tbp_id, bbp_id])]
prompt_df["prompt"] = prompt_df["prompt"].str.replace("Text: ", "")
# %%
# ------------------ FORMAT PROMPTS ------------------
# TODO: does this need to be edited for llama? If so,
# add an argument for platform
formatted_prompts = []
for prompt_text in prompt_df["prompt"]:
    message = format_prompts.format_system_message(prompt_text)
    formatted_prompts.append([message])
# %%
# ------------------ COLLECT RESPONSES ------------------# ------------------ COLLECT RESPONSES ------------------
response_rows = []
for baseline_prompt_id, prompt in zip(prompt_df.baseline_prompt_id, formatted_prompts):
    prompt_text = prompt[0]["content"]  # TODO: is this the same format for llama?

    for text, participant_id, study, question, human_code in tqdm(
        zip(df.text, df.participant_id, df.study, df.question, df.human_code),
        total=len(df),
        desc=f"Prompt ID: {baseline_prompt_id}",
    ):
        timestamp = datetime.now().isoformat()
        cleaned_response, system_fingerprint = classify.format_message_and_get_response(
            model_provider=start.PLATFORM,
            prompt=prompt,
            text_to_classify=text,
            temperature=0.0001,
        )
        classification = classify.create_binary_classification_from_response(
            cleaned_response
        )

        response_rows.append(
            {
                "participant_id": participant_id,
                "study": study,
                "question": question,
                "text": text,
                "human_code": human_code,
                "response": cleaned_response,
                "classification": classification,
                "prompt": prompt_text,
                "model": start.MODEL,
                "fingerprint": system_fingerprint,
                "baseline_prompt_id": baseline_prompt_id,
                "timestamp": timestamp,
            }
        )

long_df = pd.DataFrame(response_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
# ------------------ CREATE RESULTS FILE ------------------
if not os.path.exists(EXPORT_RESULTS_PATH):
    wb = Workbook()
    wb.save(EXPORT_RESULTS_PATH)

# ------------------ WRITE METRICS ------------------
wb = load_workbook(EXPORT_RESULTS_PATH)
if "results" in wb.sheetnames:
    del wb["results"]
ws = wb.create_sheet("results")

headers = [
    "Baseline Prompt ID",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Accuracy SE",
    "Precision SE",
    "Recall SE",
    "F1 SE",
    "Prompt",
]
for col, header in enumerate(headers, 1):
    ws.cell(row=1, column=col, value=header)

row = 2
for bp_id in long_df.baseline_prompt_id.unique():
    combo_df = long_df[long_df.baseline_prompt_id == bp_id]
    full_df = combo_df.copy()

    # Full metrics
    valid = full_df.dropna(subset=["human_code", "classification"])
    accuracy, precision, recall, f1 = classify.print_and_save_metrics(
        valid["human_code"], valid["classification"]
    )

    y_true = valid["human_code"]
    y_pred = valid["classification"]
    _, acc_se = metric_standard_errors.bootstrap_accuracy(y_true, y_pred, 1000, 12)
    _, prec_se = metric_standard_errors.bootstrap_precision(y_true, y_pred, 1000, 12)
    _, rec_se = metric_standard_errors.bootstrap_recall(y_true, y_pred, 1000, 12)
    _, f1_se = metric_standard_errors.bootstrap_f1(y_true, y_pred, 1000, 12)

    prompt_text = combo_df["prompt"].iloc[0]
    results = [
        bp_id,
        accuracy,
        precision,
        recall,
        f1,
        acc_se,
        prec_se,
        rec_se,
        f1_se,
        prompt_text,
    ]

    for col, val in enumerate(results, 1):
        ws.cell(
            row=row, column=col, value=round(val, 3) if isinstance(val, float) else val
        )
    row += 1

wb.save(EXPORT_RESULTS_PATH)
# %%
