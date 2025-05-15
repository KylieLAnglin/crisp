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

negative_examples = pd.read_excel(
    start.DATA_DIR + "temp/ncb_fewshot_example_performance_negative.xlsx"
)
negative_examples["human_code"] = 0
positive_examples = pd.read_excel(
    start.DATA_DIR + "temp/ncb_fewshot_example_performance_positive.xlsx"
)
positive_examples["human_code"] = 1
# %%
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
PROMPT_FILE = "ncb_fewshot_prompt.xlsx"
prompt_components = pd.read_excel(
    start.DATA_DIR + "prompts/" + PROMPT_FILE, sheet_name="components"
)

PROMPT_TEXT = prompt_components.loc[0]["text"]

# %%
baseline_prompt = [
    {"role": "system", "content": PROMPT_TEXT},
]

list_of_prompts = []

positive_example_text = positive_examples.iloc[0]["example_text"]
negative_example_text = negative_examples.iloc[0]["example_text"]

formatted_prompt = baseline_prompt + [
    {"role": "user", "content": positive_example_text},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": negative_example_text},
    {"role": "assistant", "content": "No"},
]
list_of_prompts.append(formatted_prompt)

# %%
example_max = 1

formatted_prompt = baseline_prompt
for i in range(example_max, -1, -1):
    positive_example_text = positive_examples.iloc[i]["example_text"]
    negative_example_text = negative_examples.iloc[i]["example_text"]
    formatted_prompt = formatted_prompt + [
        {"role": "user", "content": positive_example_text},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": negative_example_text},
        {"role": "assistant", "content": "No"},
    ]
list_of_prompts.append(formatted_prompt)

# %%
example_max = 2
formatted_prompt = baseline_prompt
for i in range(example_max, -1, -1):
    positive_example_text = positive_examples.iloc[i]["example_text"]
    negative_example_text = negative_examples.iloc[i]["example_text"]
    formatted_prompt = formatted_prompt + [
        {"role": "user", "content": positive_example_text},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": negative_example_text},
        {"role": "assistant", "content": "No"},
    ]
list_of_prompts.append(formatted_prompt)

example_max = 3
formatted_prompt = baseline_prompt

for i in range(example_max, -1, -1):
    positive_example_text = positive_examples.iloc[i]["example_text"]
    negative_example_text = negative_examples.iloc[i]["example_text"]
    formatted_prompt = formatted_prompt + [
        {"role": "user", "content": positive_example_text},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": negative_example_text},
        {"role": "assistant", "content": "No"},
    ]
list_of_prompts.append(formatted_prompt)


example_max = 4
formatted_prompt = baseline_prompt

for i in range(example_max, -1, -1):
    positive_example_text = positive_examples.iloc[i]["example_text"]
    negative_example_text = negative_examples.iloc[i]["example_text"]
    formatted_prompt = formatted_prompt + [
        {"role": "user", "content": positive_example_text},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": negative_example_text},
        {"role": "assistant", "content": "No"},
    ]
list_of_prompts.append(formatted_prompt)


example_max = 5
formatted_prompt = baseline_prompt

for i in range(example_max, -1, -1):
    positive_example_text = positive_examples.iloc[i]["example_text"]
    negative_example_text = negative_examples.iloc[i]["example_text"]
    formatted_prompt = formatted_prompt + [
        {"role": "user", "content": positive_example_text},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": negative_example_text},
        {"role": "assistant", "content": "No"},
    ]
list_of_prompts.append(formatted_prompt)

# %%
positive_examples_indices = positive_examples.head(6).example_index.values
negative_examples_indices = negative_examples.head(6).example_index.values

# combine positive and negative examples
combined_examples = pd.concat([positive_examples, negative_examples])

# drop examples indices
test_df = combined_examples.set_index("example_index").drop(
    np.concatenate((positive_examples_indices, negative_examples_indices))
)
example_performance_dicts = []
for formatted_prompt in list_of_prompts:
    example_performance_dict = {}
    example_performance_dict["num_examples"] = len(formatted_prompt) // 4
    responses = []
    classifications = []
    for text in test_df.example_text:
        messages = formatted_prompt + [{"role": "user", "content": text}]

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

    example_performance_dicts.append(example_performance_dict)

# %%
example_performance_df = pd.DataFrame(example_performance_dicts)
example_performance_df = example_performance_df.sort_values("f1", ascending=False)
example_performance_df.to_excel(
    start.DATA_DIR + "temp/ncb_fewshot_num_example_performance.xlsx"
)
