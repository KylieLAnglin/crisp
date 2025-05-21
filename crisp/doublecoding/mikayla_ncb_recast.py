# %%
import pandas as pd
import numpy as np
from crisp.library import start

df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs_original_gold.xlsx")

df = df[df.study == "recast"]
# %%
# %%
# take random sample of 30 emotion question rows
recast_emotion_sample = df[df.question == "emotion"].sample(30, random_state=1)
recast_attachment_sample = df[df.question == "attachment"].sample(30, random_state=1)
recast_identity = df[df.question == "identity"]

# append the samples, create a new dataframe
recast_sample_for_double_coding = pd.concat(
    [recast_emotion_sample, recast_attachment_sample, recast_identity],
    ignore_index=True,
)

recast_sample_for_double_coding["self"] = ""
recast_sample_for_double_coding["other"] = ""
recast_sample_for_double_coding["trust"] = ""
recast_sample_for_double_coding["notes"] = ""

recast_sample_for_double_coding = recast_sample_for_double_coding[
    ["participant_id", "question", "self", "other", "trust", "notes", "text"]
]
recast_sample_for_double_coding.to_excel(
    start.DATA_DIR + "temp/for_coders/doublecoding_ncb_recast.xlsx", index=False
)
