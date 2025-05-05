# %%
import shutil
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from crisp.library import start
from crisp.library import secrets, classify, export_results

from tqdm import tqdm

# %% Text file
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]

task_prompt = "Do you think this text contains an example of the psychological concept of a negative core belief? Respond Yes or No. Text: "

base_message = [{"role": "system", "content": task_prompt}]


# %%
text0 = "I was 15 almost 16 visiting my dream school. We were staying at my cousins, where we stayed for Thanksgiving every year. He was probably near 23 years old at the time. He supplied alcohol while we watched a movie in his bed. I did not expect anything to happen, as we’d grown up together. I remember being frozen after he raped me and him dressing me and carrying me to my own room. I saw him briefly the next day in the family room while both sets of parents were present. He left to go to a friends. I couldn’t tell anyone because it would tear the family apart. I have not connected with them since despite the whole family to holidays and such. I block him out of my life as much as possible. It feels like a lifetime ago that this happened but I will never have a relationship with him or his family again as a result. My sister will find out in time. This event had a negative impact directly after. I failed classes. I acted out. And I was suicidal. I entered therapy but never discussed it because I was too shameful and afraid of the consequences. I had since grown and reacted and blame only him. I was a child and he was a predatory pedophile. I feel no need to reconcile a relationship with that branch of my family. I do not think about it anymore, except in the rare occasion its triggered."
text1 = "I do not have a relationship with anyone on earth actually. I am really a hot head right now. My parents have controlled me and manipulated me and I don't have a way out. I believe that everyone is probably evil and I am just expecting to get yelled at all the time. Every conversation I hear someone have I expect it to end in yelling. I have to think about my every move, walking, opening a door just waiting to get yelled at. I have been led to believe I am a bad person and I can never have any relationship. And I really can't because I expect anyone to get mad at me or yell at me or I am inconveniencing them and everything I do is wrong."

messages = base_message + [
    {"role": "user", "content": text0},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": text1},
    {"role": "assistant", "content": "Yes"},
]

# %%
responses = []
classifications = []
for text in tqdm(df.text):
    cleaned_response = classify.format_message_and_get_response(
        prompt=messages,
        text_to_classify=text,
    )
    responses.append(cleaned_response)
    classification = classify.create_binary_classification_from_response(
        cleaned_response
    )
    classifications.append(classification)
response_df = pd.DataFrame(
    {
        "text": df.text,
        "human_codes": df.human_code,
        "response": responses,
        "classification": classifications,
    }
)
# %%
classify.print_and_save_metrics(response_df.human_codes, response_df.classification)
