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
num_examples = 5
negative_examples = negative_examples.head(num_examples)
positive_examples = positive_examples.head(num_examples)
examples = pd.concat([positive_examples, negative_examples])
examples["gold_standard"] = np.where(examples.human_code == 1, "Yes", "No")
# create a list of dataframes, each with a random order of examples, but need a seed
example_dfs = []
for i in range(10):
    example_df = examples.sample(frac=1, random_state=i)
    example_df["order"] = i
    example_df = example_df.reset_index(drop=True)
    example_dfs.append(example_df)
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
for i in range(10):
    formatted_prompt = baseline_prompt
    for example_index in example_dfs[i].index:
        example_text = example_dfs[i].loc[example_index]["example_text"]
        assistant_response = example_dfs[i].loc[example_index]["gold_standard"]
        formatted_prompt = formatted_prompt + [
            {"role": "user", "content": example_text},
            {"role": "assistant", "content": assistant_response},
        ]
    list_of_prompts.append(formatted_prompt)


# %%
test_df = df.drop(examples.example_index)
example_performance_dicts = []
for formatted_prompt in list_of_prompts:
    example_performance_dict = {}
    example_performance_dict["num_examples"] = len(formatted_prompt) // 4
    responses = []
    classifications = []
    for text in test_df.text:
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
example_performance_df["df_num"] = example_performance_df.index
example_performance_df = example_performance_df.sort_values("f1", ascending=False)
example_performance_df.to_excel(
    start.DATA_DIR + "temp/ncb_fewshot_order_example_performance.xlsx"
)


# %%
# id df_num of top f1
top_df_num = int(example_performance_df.iloc[0]["df_num"])
top_df = example_dfs[top_df_num]

top_df[["example_text", "human_code", "gold_standard"]].to_excel(
    start.DATA_DIR + "temp/ncb_fewshot_examples.xlsx"
)

# %%
