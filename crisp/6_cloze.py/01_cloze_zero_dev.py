# 1_baseline_prompt/01_cloze_zero_dev.py
# %%
import os
import pandas as pd
from tqdm import tqdm

from crisp.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

print(f"Running cloze-style zero-shot evaluation on {CONCEPT} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_cloze_zero_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cloze_zero_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ DEFINE PROMPT TEMPLATE ------------------
PROMPT_TEMPLATE = (
    "{text}\n\n"
    "Overall, I now feel negative about things. "
    "Please complete the sentence by choosing either 'only sometimes' or 'a lot'.\n"
    "Answer:"
)


def create_prompt(text):
    return PROMPT_TEMPLATE.format(text=text.strip())


# ------------------ EVALUATE ------------------
response_rows = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating cloze completions"):
    prompt = create_prompt(row["text"])

    result = classify.evaluate_prompt(
        prompt_text=prompt,
        prompt_id="cloze_zero",
        df=pd.DataFrame([row]),  # evaluate one row at a time
        platform=PLATFORM,
        temperature=0.0,
        completion_only=True,  # model only returns the completion
    )[0]

    # Normalize and extract predicted label
    completion = result["response"].strip().lower()

    if "a lot" in completion:
        predicted_label = 1
    elif "only sometimes" in completion:
        predicted_label = 0
    else:
        predicted_label = None  # fallback for unexpected output

    result["prompt_id"] = "cloze_zero"
    result["prompt"] = prompt
    result["pred"] = predicted_label
    result["true"] = row["human_code"]
    response_rows.append(result)

# ------------------ EXPORT ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(RESPONSE_PATH, index=False)

classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)
