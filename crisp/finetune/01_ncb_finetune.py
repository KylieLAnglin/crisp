# %%
import shutil
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import json
import jsonlines
from crisp.library import start
from crisp.library import secrets

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# %% Text file
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
df["gold_standard"] = np.where(df.human_code == 1, "Yes", "No")
df["text"] = df.text.str.replace("\n", " ")
# remove quotation marks from text
df["text"] = df.text.str.replace('"', "")
# %% Import prompts
PROMPT_TEXT = "Negative core beliefs are generalized or exaggerated judgments people have about themselves, others, or the world. They are often inaccurate, harmful, or just not useful. The key idea is that negative core beliefs are more generalized than warranted. Negative core beliefs usually come in the form of absolutist or all-or-none language like: I am, people are, the world is, everybody, nobody, always, and never, with few qualifiers. Again, negative core beliefs are overly generalized negative statements. Does the following text contain an example of the psychological concept of a negative core belief? Respond Yes or No. Text:"
jsonl_path = start.DATA_DIR + "clean/ncb_finetune.jsonl"

# %% Fine tune with training examples in df.text and df.human_code
formatted_prompt = {
    "role": "system",
    "content": PROMPT_TEXT,
}
examples = []
for text, classification in zip(df.text, df.gold_standard):
    message_list = []
    message_list.append(formatted_prompt)
    message_list.append({"role": "user", "content": text})
    message_list.append({"role": "assistant", "content": classification})
    examples.append({"messages": message_list})

with jsonlines.open(jsonl_path, mode="w") as writer:
    for example in examples:
        writer.write(example)

# %%
# job = client.fine_tuning.jobs.create(
#     training_file=jsonl_path,
#     model="gpt-4o-2024-08-06",
#     method={
#         "type": "dpo",
#         "dpo": {
#             "hyperparameters": {"beta": 0.1},
#         },
#     },
# )
# client = OpenAI()

# # %%
# from openai import OpenAI

# client = OpenAI()

# job = client.fine_tuning.jobs.create(
#     training_file="file-all-about-the-weather",
#     model="gpt-4o-2024-08-06",
#     method={
#         "type": "dpo",
#         "dpo": {
#             "hyperparameters": {"beta": 0.1},
#         },
#     },
# )

# from openai import OpenAI

# client = OpenAI()

# client.fine_tuning.jobs.create(
#     training_file="file-abc123", model="gpt-4o-mini-2024-07-18"
# )
