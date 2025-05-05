# %%
import shutil
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from crisp.library import export_results
from crisp.library import format_prompts
from crisp.library import classify
from crisp.library import start
from crisp.library import secrets

OPENAI_API_KEY = secrets.OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_FILE = "ncb_main_prompts.xlsx"
# Read the prompts from the Excel file
prompts = pd.read_excel(start.DATA_DIR + "prompts/" + PROMPT_FILE, sheet_name=None)

# Read the text data from the Excel file
text_df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
text_df = text_df[text_df.split_group == "dev"]
client = OpenAI(api_key=secrets.OPENAI_API_KEY)
MODEL = "ft:gpt-4o-2024-08-06:personal:ncb-2025-03-22:BDpyRcZ2"
# %%
# Specify the desired prompt name, or leave empty for all
SHEET_NAME = "optimized_baseline"

# Iterate over each sheet in the prompts Excel file
for sheet_name, sheet_data in prompts.items():
    if ((sheet_name == SHEET_NAME) | (SHEET_NAME == "")) & ("Sheet" not in sheet_name):
        print(sheet_name)

        # get prompt
        prompt = []
        for message in sheet_data.index:
            dict_entry = {
                "role": sheet_data.loc[message, "role"],
                "content": sheet_data.loc[message, "content"],
            }
            prompt.append(dict_entry)
        # send each row of text to the model and extract response
        responses = []
        classifications = []
        for text in tqdm(text_df.text):
            messages = prompt + [{"role": "user", "content": text}]

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.00,
            )
            # clean and append response
            cleaned_response = response.choices[0].message.content
            responses.append(cleaned_response)
            classification = classify.create_binary_classification_from_response(
                cleaned_response
            )
            classifications.append(classification)

        text_df["prompt"] = sheet_name
        text_df["response"] = responses
        text_df["classification"] = classifications
        text_df["agreement"] = np.where(text_df.response == text_df.human_code, 1, 0)
        response_df = pd.DataFrame(
            {
                "participant_id": text_df.participant_id,
                "study": text_df.study,
                "question": text_df.question,
                "text": text_df.text,
                "human_code": text_df.human_code,
                "response": responses,
                "classification": classifications,
                "agreement": text_df.agreement,
            }
        )

        with pd.ExcelWriter(
            start.MAIN_DIR + f"data/clean/classifications_ncb_dev.xlsx",
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",  # Add this line to replace the sheet if it already exists
        ) as writer:
            text_df.to_excel(writer, sheet_name="finetuned", index=False)

# %%
