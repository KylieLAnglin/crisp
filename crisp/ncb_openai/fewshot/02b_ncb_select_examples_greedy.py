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
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
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

# Baseline system prompt
baseline_prompt = [
    {
        "role": "system",
        "content": PROMPT_TEXT,
    }
]

# Store results
results = []
NUM_EXAMPLES = 6


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


def evaluate_fewshot(candidate_examples, test_data):
    """Evaluates a given few-shot example set on the test data and returns the F1-score."""
    fewshot_prompt = generate_fewshot_prompt(candidate_examples)
    predictions = []

    for _, row in test_data.iterrows():
        prediction = get_openai_response(client, fewshot_prompt, row.text)
        binary_prediction = 1 if prediction.lower() == "yes" else 0
        predictions.append(binary_prediction)

    return f1_score(test_data.human_code, predictions)


def greedy_fewshot_selection(df):
    """Selects the best set of 6 examples using a greedy search approach."""
    candidate_pool = df[df.split_group == "train"].copy()
    selected_examples = pd.DataFrame()
    test_data = (
        candidate_pool.copy()
    )  # The rest of the training data is used as the test set

    with tqdm(total=NUM_EXAMPLES, desc="Greedy Few-Shot Selection") as pbar:
        for _ in range(NUM_EXAMPLES):
            best_example = None
            best_f1 = -1

            for _, candidate in candidate_pool.iterrows():
                temp_set = pd.concat([selected_examples, candidate.to_frame().T])
                f1 = evaluate_fewshot(temp_set, test_data)

                if f1 > best_f1:
                    best_f1 = f1
                    best_example = candidate

            selected_examples = pd.concat(
                [selected_examples, best_example.to_frame().T]
            )
            candidate_pool = candidate_pool.drop(best_example.name)

            pbar.update(1)

    return selected_examples


# Run the greedy selection
best_examples = greedy_fewshot_selection(df)
best_examples.to_csv(start.RESULTS_DIR + "ncb_greedy_fewshot_samples.csv", index=False)

print("Greedy search complete. Best examples saved to ncb_greedy_fewshot_samples.csv.")
