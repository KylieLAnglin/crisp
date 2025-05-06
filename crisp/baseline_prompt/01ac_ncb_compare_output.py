# %%

import pandas as pd
from openai import OpenAI
from openpyxl import load_workbook
from tqdm import tqdm
import numpy as np

from crisp.library import start
from crisp.library import secrets
from crisp.library import export_results
from crisp.library import format_prompts
from crisp.library import classify


FILE1 = "ncb_variants_responses 2025-05-06 144 pm"
FILE2 = "ncb_variants_responses 2025-05-06 316 pm"
DROP_VARS = ["response2", "response3", "classification2", "classification3"]
unique_vars = [
    "response",
    "classification",
    "fingerprint",
]
long_df1 = pd.read_excel(start.DATA_DIR + "temp/" + FILE1 + ".xlsx").drop(
    columns=DROP_VARS
)
long_df1 = long_df1.rename(
    columns={
        "response": "response_a",
        "classification": "classification_a",
        "fingerprint": "fingerprint_a",
    }
)
long_df2 = pd.read_excel(start.DATA_DIR + "temp/" + FILE2 + ".xlsx")
long_df2 = long_df2.drop(columns=DROP_VARS)
long_df2 = long_df2.rename(
    columns={
        "response": "response_b",
        "classification": "classification_b",
        "fingerprint": "fingerprint_b",
    }
)

# %%

# %%
identifying_vars = [
    "participant_id",
    "study",
    "question",
    "text",
    "human_code",
    "prompt",
    "model",
    "combo_id",
]

long_df = long_df1.merge(long_df2, how="inner", on=identifying_vars)
long_df = long_df[
    identifying_vars
    + [
        "response_a",
        "response_b",
        "classification_a",
        "classification_b",
        "fingerprint_a",
        "fingerprint_b",
    ]
]

long_df["agree"] = np.where(
    long_df["classification_a"] == long_df["classification_b"], 1, 0
)
long_df.to_excel(
    start.DATA_DIR + "temp/ncb_variants_responses_replicated.xlsx", index=False
)
# %%
# %%


long_df.agree.value_counts()

# %%
long_df.human_code.value_counts()

long_df[long_df.human_code == 1].agree.value_counts()
