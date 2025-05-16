# %%
# %%
from crisp.library import start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df1 = pd.read_excel(start.DATA_DIR + "temp/meaning_making.xlsx")
df1["construct"] = "meaning_making"
df1 = df1[
    [
        "participant_id",
        "study",
        "question",
        "text",
        "construct",
        "meaning_making",
    ]
]
df1 = df1.rename(columns={"meaning_making": "human_code"})
df1 = df1[df1.human_code.notna()]

# %%
df2 = pd.read_excel(start.DATA_DIR + "temp/negative_core_beliefs.xlsx")
df2["construct"] = "negative_core_beliefs"
df2 = df2[
    [
        "participant_id",
        "study",
        "question",
        "text",
        "construct",
        "negative_belief_any",
    ]
]
df2 = df2.rename(columns={"negative_belief_any": "human_code"})
df2 = df2[df2.human_code.notna()]

# %%
df = pd.concat([df1, df2])
df.head()
#
df["unique_text_id"] = (
    df["participant_id"].astype(str) + "_" + df["study"] + "_" + df["question"]
)

# check that text is unique within unique text id
# It's not because there are different constructs
# assert df["text"].nunique() == df["unique_text_id"].nunique()

# one duplicate within text id and construct
df[
    df.duplicated(subset=["unique_text_id", "text", "construct"], keep=False)
].sort_values(by="unique_text_id")


participant_df = df[["participant_id", "study"]].drop_duplicates(
    subset=["participant_id", "study"]
)

# drop if missing participant_id or study
participant_df = participant_df.dropna(subset=["participant_id", "study"])

# assign random number from 1 to 1000 for each participant
np.random.seed(1)  # 1001
participant_df["random_number"] = np.random.randint(1, 100001, participant_df.shape[0])
participant_df = participant_df.sort_values(by=["random_number"])

# %%
# first 25% of participants are train
participant_df["order"] = np.arange(1, participant_df.shape[0] + 1)

n = participant_df.order.max()
train_size = int(n * 0.25)
dev_size = int(n * 0.5)
test_size = int(n * 0.25)
participant_df["train"] = participant_df.order <= train_size
participant_df["dev"] = (participant_df.order > train_size) & (
    participant_df.order <= train_size + dev_size
)
participant_df["test"] = participant_df.order > train_size + dev_size

participant_df["split_group"] = np.where(participant_df.train, "train", "")
participant_df["split_group"] = np.where(
    participant_df.dev, "dev", participant_df["split_group"]
)
participant_df["split_group"] = np.where(
    participant_df.test, "test", participant_df["split_group"]
)
# %%
participant_df.to_excel(start.DATA_DIR + "clean/participant_split.xlsx", index=False)

# %%
final_df = df.merge(participant_df, on=["participant_id", "study"], how="inner")

final_df[final_df.construct == "meaning_making"].to_excel(
    start.DATA_DIR + "clean/meaning_making.xlsx", index=False
)
final_df[final_df.construct == "negative_core_beliefs"].to_excel(
    start.DATA_DIR + "clean/ncb.xlsx", index=False
)

# %%
# Bar graph, number of pos/neg ncg examples in train, dev, test
temp_df = final_df[final_df.construct == "negative_core_beliefs"]
plt.figure(figsize=(10, 6))
temp_df.groupby(["split_group", "human_code"]).size().unstack().plot(kind="bar")

plt.title("Number of Positive and Negative Negative Core Beliefs in Train, Dev, Test")
plt.ylabel("Number of Examples")
plt.xlabel("Split Group")
plt.xticks(rotation=0)
plt.show()
# sort train dev test

# %%
# print proportion of positive and negative examples in each split
temp_df = final_df[final_df.construct == "negative_core_beliefs"]
temp_df = temp_df.groupby(["split_group", "human_code"]).size().unstack()
temp_df = temp_df.fillna(0)
temp_df = temp_df.div(temp_df.sum(axis=1), axis=0)
temp_df = temp_df.reset_index()
temp_df = temp_df.rename(columns={0: "negative", 1: "positive"})
temp_df  # %%

# %%
