# %%
import shutil
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from crisp.library import secrets

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def format_message_and_get_response(
    prompt, text_to_classify, model="gpt-4o", temperature=0.00
):
    messages = prompt + [{"role": "user", "content": text_to_classify}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    cleaned_response = response.choices[0].message.content
    return cleaned_response


def create_binary_classification_from_response(response):
    if "yes" in response.lower():
        classification = 1
    elif "no" in response.lower():
        classification = 0
    else:
        classification = np.nan
    return classification


def print_and_save_metrics(human_codes, classifications):
    accuracy = accuracy_score(human_codes, classifications)
    precision = precision_score(human_codes, classifications)
    recall = recall_score(human_codes, classifications)
    f1 = f1_score(human_codes, classifications)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    return accuracy, precision, recall, f1
