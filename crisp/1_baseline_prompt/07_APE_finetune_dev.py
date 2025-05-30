# 08_baseline_finetune_dev.py
import pandas as pd
from tqdm import tqdm
from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

print(f"Running fine-tuned model eval on {CONCEPT} with {PLATFORM} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)
MODEL_NAME_PATH = (
    start.DATA_DIR
    + f"finetune/{PLATFORM}_{CONCEPT}_baseline_finetuned_model_names.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_baseline_finetuned_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_finetuned_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD PROMPTS ------------------
train_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

prompt_map = {
    "top": train_results.loc[train_results["prompt_id"] == top_id, "prompt"].values[0],
    "bottom": train_results.loc[
        train_results["prompt_id"] == bottom_id, "prompt"
    ].values[0],
}

# ------------------ LOAD FINE-TUNED MODEL NAMES ------------------
model_names_df = pd.read_excel(MODEL_NAME_PATH)
model_map = dict(
    zip(model_names_df["category"], model_names_df["fine_tuned_model_name"])
)

# ------------------ GENERATE RESPONSES ------------------
response_rows = []

for label in ["top", "bottom"]:
    prompt_text = prompt_map[label].replace("Text:", "").strip()
    model_name = model_map.get(label)

    print(f"Evaluating {label} model: {model_name}")

    responses = classify.evaluate_prompt(
        prompt_text=prompt_text,
        prompt_id=label,
        df=df,
        platform=PLATFORM,
        temperature=0.0,
    )
    response_rows.extend(responses)

# ------------------ SAVE RESPONSES ------------------
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
