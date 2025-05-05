# %%
# %%
import shutil
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from crisp.library import start
from crisp.library import secrets
from tqdm import tqdm
import random
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score


OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
NUM_SAMPLES = 100
NUM_EXAMPLES = 6
# %% Text file
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
df["gold_code"] = np.where(df.human_code == 1, "Yes", "No")
PROMPT_FILE = "ncb_fewshot_prompt.xlsx"
prompt_components = pd.read_excel(
    start.DATA_DIR + "prompts/" + PROMPT_FILE, sheet_name="components"
)

PROMPT_TEXT = prompt_components.loc[0]["text"]

baseline_prompt = [
    {
        "role": "system",
        "content": PROMPT_TEXT,
    }
]
###
#
###

import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

# Number of random samples
NUM_SAMPLES = 100
NUM_EXAMPLES = 6

# Baseline system prompt
baseline_prompt = [
    {
        "role": "system",
        "content": PROMPT_TEXT,
    }
]

# Store results
results = []


def generate_fewshot_prompt(sampled_examples):
    """Formats the selected few-shot examples into chat-style format."""
    fewshot_pairs = []
    for _, row in sampled_examples.iterrows():
        fewshot_pairs.append({"role": "user", "content": f"Text: {row.text}"})
        fewshot_pairs.append({"role": "assistant", "content": row.gold_code})
    return fewshot_pairs


def get_openai_response(client, fewshot_pairs, new_text):
    """Sends the prompt to OpenAI and retrieves the classification response."""
    prompt = (
        baseline_prompt
        + fewshot_pairs
        + [{"role": "user", "content": f"Text: {new_text}"}]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=prompt, max_tokens=5  # Limit output length
    )
    return response.choices[0].message.content.strip()


# Run 100 random samples
with tqdm(total=NUM_SAMPLES, desc="Optimizing Few-Shot Selection") as pbar:
    for _ in range(NUM_SAMPLES):
        sampled_examples = df[df.split_group == "train"].sample(
            n=NUM_EXAMPLES, random_state=random.randint(0, 9999)
        )
        fewshot_prompt = generate_fewshot_prompt(sampled_examples)

        predictions = []
        test_data = df[df.split_group == "train"].drop(
            sampled_examples.index
        )  # Ensure test data is separate

        for _, row in test_data.iterrows():
            prediction = get_openai_response(client, fewshot_prompt, row.text)
            binary_prediction = 1 if prediction.lower() == "yes" else 0
            predictions.append(binary_prediction)

        # Evaluate performance
        f1 = f1_score(test_data.human_code, predictions)
        results.append(
            {
                "sample_id": _,
                "f1": f1,
                "examples": sampled_examples.unique_text_id.tolist(),
            }
        )

        pbar.update(1)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(start.RESULTS_DIR + "ncb_optimal_fewshot_samples.csv", index=False)

print("Optimization complete. Results saved to optimal_fewshot_samples.csv.")

# %%
