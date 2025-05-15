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

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# %% Text file
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
df["gold_code"] = np.where(df.human_code == 1, "Yes", "No")
PROMPT_FILE = "ncb_fewshot_prompt.xlsx"
prompt_components = pd.read_excel(
    start.DATA_DIR + "prompts/" + PROMPT_FILE, sheet_name="components"
)

PROMPT_TEXT = prompt_components.loc[0]["text"]

# %%
# split into 5 groups of exactly 20, one with 21
df = df.sort_values(by="random_number")
df["group"] = np.nan
df.loc[df.index[:20], "group"] = 1
df.loc[df.index[20:40], "group"] = 2
df.loc[df.index[40:60], "group"] = 3
df.loc[df.index[60:80], "group"] = 4
df.loc[df.index[80:], "group"] = 5

df.group.value_counts()
df.text.nunique()
# %%

baseline_prompt = [
    {
        "role": "system",
        "content": PROMPT_TEXT,
    }
]

# use cross validation to determine best number of examples
# for each group select 1-10 random examples
# get responses other folds
# calculate average performance via f1
# save performance for each group and number of examples


def generate_fewshot_examples(df, group, num_examples):
    """Selects num_examples from a given group and formats them as chat-style examples."""
    examples = df[df.group == group].sample(n=num_examples, random_state=42)
    fewshot_pairs = []

    for _, row in examples.iterrows():
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
        model="gpt-4o",  # Adjust model as needed
        messages=prompt,
        max_tokens=5,  # Small limit since the response is binary
    )
    return response.choices[0].message.content.strip()


results = []

total_iterations = 10 * 5  # 10 num_examples * 5 test groups
with tqdm(total=total_iterations, desc="Cross-validation Progress") as pbar:
    for num_examples in range(1, 11):  # Trying 1 to 10 examples
        for test_group in range(1, 6):  # Each group takes a turn as the test set
            train_groups = [g for g in range(1, 6) if g != test_group]
            train_data = df[df.group.isin(train_groups)]
            test_data = df[df.group == test_group]

            # Generate few-shot examples
            fewshot_prompt = generate_fewshot_examples(
                train_data, random.choice(train_groups), num_examples
            )

            # Get predictions
            predictions = []
            for _, row in test_data.iterrows():
                prediction = get_openai_response(client, fewshot_prompt, row.text)
                binary_prediction = 1 if prediction.lower() == "yes" else 0
                predictions.append(binary_prediction)

            # Evaluate performance
            f1 = f1_score(test_data.human_code, predictions)
            results.append(
                {"num_examples": num_examples, "test_group": test_group, "f1": f1}
            )

            pbar.update(1)
# %%
# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(start.RESULTS_DIR + "ncb_fewshot_cv_results.csv", index=False)

print("Cross-validation complete. Results saved to ncb_fewshot_cv_results.csv.")

# %%
results_summary = (
    results_df[["num_examples", "f1"]]
    .groupby("num_examples")
    .agg(["mean", "min", "max"])
)
results_summary = results_summary.round(2)
print(results_summary)
results_df
