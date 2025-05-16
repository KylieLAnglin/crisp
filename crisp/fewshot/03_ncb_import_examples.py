# %%

import os
import ast
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

# %%
example_indices_df = pd.read_csv(start.RESULTS_DIR + "ncb_optimal_fewshot_samples.csv")
example_indices_df = example_indices_df.sort_values("f1", ascending=False)
# %%
example_indices_df

top_examples = example_indices_df.head(1)["examples"]
# %%
import ast

top1_loc = 10
top2_loc = 59
top_examples_str = example_indices_df.loc[top1_loc][
    "examples"
]  # Get the first row's examples as a string
top_examples = ast.literal_eval(top_examples_str)  # Convert string to list
# %%
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
df["gold_code"] = np.where(df.human_code == 1, "Yes", "No")

examples = df[df.unique_text_id.isin(top_examples)]
examples.to_excel(start.DATA_DIR + "clean/ncb_top_examples.xlsx", index=False)
# %%


# %%
tests = df[~df.unique_text_id.isin(top_examples)]

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

client = OpenAI(api_key=secrets.OPENAI_API_KEY)
MODEL = "gpt-4o"
# %%
# Specify the desired prompt name, or leave empty for all
SHEET_NAME = "optimized_fewshot"

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
# %%
# send each row of text to the model and extract response

# %%
responses = []
classifications = []
for text in tqdm(tests.text):
    cleaned_response = classify.format_message_and_get_response(
        prompt=prompt,
        text_to_classify=text,
    )
    responses.append(cleaned_response)
    classification = classify.create_binary_classification_from_response(cleaned_response)
    classifications.append(classification)
response_df = pd.DataFrame(
    {
        "text": tests.text,
        "human_codes": tests.human_code,
        "response": responses,
        "classification": classifications,
    }
)
# %%
classify.print_and_save_metrics(response_df.human_codes, response_df.classification)
