# %%
from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from crisp.library import start
from crisp.library import secrets
from crisp.library import classify

# ------------------ CONSTANTS ------------------
MODEL = start.MODEL
OPENAI_API_KEY = secrets.OPENAI_API_KEY
PERSONAS = [
    "You are a brilliant psychologist.",
    "You are an excellent therapist.",
    "You are a very smart mental health researcher.",
]
PROMPT_FILE = "ncb_variants"
PROMPT_PATH = start.MAIN_DIR + "results/" + PROMPT_FILE + "_results_dev.xlsx"
OUTPUT_TRACKING_FILE = start.DATA_DIR + "temp/ncb_persona_training_results.csv"

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------ LOAD PROMPTS AND TRAINING DATA ------------------
prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="combo_results")
best_prompt = prompt_df.loc[prompt_df["F1"].idxmax(), "Prompt"]
worst_prompt = prompt_df.loc[prompt_df["F1"].idxmin(), "Prompt"]

df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
df = df[df.text.notna() & df.human_code.notna()]
df = df.rename(columns={"human_code": "label"})


# ------------------ EVALUATION ------------------
def apply_prompt(df, system_prompt):
    predictions = []
    for text in tqdm(df["text"], desc="Classifying texts"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.00001,
            n=1,
        )
        prediction = classify.create_binary_classification_from_response(
            response.choices[0].message.content
        )
        predictions.append(prediction)
    return predictions


records = []
for persona in PERSONAS:
    for prompt_label, base_prompt in [("best", best_prompt), ("worst", worst_prompt)]:
        full_prompt = f"{persona}\n{base_prompt}"
        preds = apply_prompt(df, full_prompt)
        f1 = f1_score(df["label"].tolist(), preds)
        records.append(
            {
                "persona": persona,
                "prompt_type": prompt_label,
                "f1_score": f1,
                "prompt": full_prompt,
            }
        )
        print(f"Persona: {persona} | Prompt: {prompt_label} | F1: {f1:.4f}")

results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUT_TRACKING_FILE, index=False)
print(f"Saved results to {OUTPUT_TRACKING_FILE}")
# %%
