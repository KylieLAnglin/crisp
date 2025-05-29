# %%
from crisp.library import start
import pandas as pd
import numpy as np

SEED = start.SEED
np.random.seed(SEED)
# %%
df = pd.read_csv(start.DATA_DIR + "raw/Emotion_classify_Data.csv")
df = df.sample(575, random_state=SEED)

# %%
# create the column human code
# Assign human_code: 1 if emotion is 'anger', else 0
df["human_code"] = (
    df["Emotion"].str.lower().apply(lambda x: 1 if x == "anger" else 0)
)  # not sure about this line of code
df["construct"] = "anger"

df = df.rename(columns={"Comment": "text"})
df["participant_id"] = df.index + 1  # or use a real ID if available, do we need this?
df["study"] = "emotion_classify"
df["question"] = "emotion"
df["unique_text_id"] = (
    df["participant_id"].astype(str) + "_" + df["study"] + "_" + df["question"]
)

df = df[
    [
        "participant_id",
        "study",
        "question",
        "text",
        "unique_text_id",
        "construct",
        "human_code",
    ]
]

# %% split
# assign random number from 1 to 1000 for each participant
np.random.seed(1)  # 1001
df["random_number"] = np.random.randint(1, 100001, df.shape[0])
df = df.sort_values(by=["random_number"])

df["order"] = np.arange(1, df.shape[0] + 1)

n = df.order.max()
train_size = int(n * 0.25)
dev_size = int(n * 0.5)
test_size = int(n * 0.25)
df["train"] = df.order <= train_size
df["dev"] = (df.order > train_size) & (df.order <= train_size + dev_size)
df["test"] = df.order > train_size + dev_size

df["split_group"] = np.where(df.train, "train", "")
df["split_group"] = np.where(df.dev, "dev", df["split_group"])
df["split_group"] = np.where(df.test, "test", df["split_group"])


df["train_use"] = pd.Series(dtype="object")

train_rows = df[df["split_group"] == "train"]

# Assign "example" to sampled train rows for few-shot training
example_sample = train_rows.sample(n=50, random_state=SEED)
df.loc[example_sample.index, "train_use"] = "example"

# Assign "eval" to remaining train rows
remaining_train_mask = (df["split_group"] == "train") & (df["train_use"].isna())
df.loc[remaining_train_mask, "train_use"] = "eval"

# Save to Excel
df.to_excel(start.DATA_DIR + f"clean/anger.xlsx", index=False)
