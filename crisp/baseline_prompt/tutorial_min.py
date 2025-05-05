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


# %% Text file
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]

text = "I think the emotional climate that I grew up in has influenced me in mostly negative ways unfortunately. I feel deeply insecure, My self esteem is almost zero. I can’t focus on tasks, and often feel very discouraged, which leads to a lack of motivation. Every day is a struggle, but every day is also a challenge I willingly accept, that I and can learn and grow from."

task_prompt = "Do you think this text contains an example of the psychological concept of a negative core belief? Respond Yes or No. Text: "

messages = [{"role": "system", "content": task_prompt}] + [
    {"role": "user", "content": text}
]

# %%

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.00,
)
