# %%
import shutil
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from crisp.library import start
from crisp.library import secrets

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# %% Text file
df = pd.read_excel(start.DATA_DIR + "clean/meaning_making.xlsx")
df = df[df.split_group == "train"]
PROMPT_FILE = "mm_fewshot_prompt.xlsx"
prompt_components = pd.read_excel(
    start.DATA_DIR + "prompts/" + PROMPT_FILE, sheet_name="components"
)

PROMPT_TEXT = prompt_components.loc[0]["text"]

# %%
prompt = {
    "role": "system",
    "content": PROMPT_TEXT,
}
# %% Format prompts
negative_example = df[df.human_code == 0].head(1).text.values[0]

formatted_prompt = [
    {"role": "system", "content": PROMPT_TEXT},
    {"role": "user", "content": negative_example},
    {"role": "assistant", "content": "No"},
]

# %%
example_performance = []

for example_index in df[df.human_code == 1].index:

    positive_example_text = df.loc[example_index]["text"]

    example_performance_dict = {}
    example_performance_dict["example_text"] = positive_example_text
    example_performance_dict["example_index"] = example_index

    new_formatted_prompt = formatted_prompt + [
        {"role": "user", "content": positive_example_text},
        {"role": "assistant", "content": "Yes"},
    ]

    # get responses from remaining examples
    test_df = df.drop(example_index)

    responses = []
    classifications = []
    for text in test_df.text:
        messages = new_formatted_prompt + [{"role": "user", "content": text}]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.00,
        )
        cleaned_response = response.choices[0].message.content
        responses.append(cleaned_response)
        if "yes" in cleaned_response.lower():
            classification = 1
        elif "no" in cleaned_response.lower():
            classification = 0
        else:
            classification = np.nan
        classifications.append(classification)

    # accuracy
    accuracy = accuracy_score(
        test_df.human_code,
        classifications,
    )
    example_performance_dict["accuracy"] = accuracy

    # precision
    precision = precision_score(
        test_df.human_code,
        classifications,
        zero_division=0,
    )
    example_performance_dict["precision"] = precision

    # recall
    recall = recall_score(
        test_df.human_code,
        classifications,
    )
    example_performance_dict["recall"] = recall

    # f1
    f1 = f1_score(
        test_df.human_code,
        classifications,
    )
    example_performance_dict["f1"] = f1

    example_performance.append(example_performance_dict)

    # %%
example_performance_df = pd.DataFrame(example_performance)
example_performance_df = example_performance_df.sort_values("f1", ascending=False)
example_performance_df.to_excel(
    start.DATA_DIR + "temp/mm_fewshot_example_performance_positive.xlsx"
)
