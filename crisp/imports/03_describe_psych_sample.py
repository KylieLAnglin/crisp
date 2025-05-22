# %%
from crisp.library import start
import pandas as pd
import numpy as np


# %%
df1 = pd.read_excel(start.DATA_DIR + "temp/meaning_making.xlsx")

# %%
for study in df1.study.value_counts().index:
    print(study)
    print(df1[df1.study == study].participant_id.nunique())

# %%
for study in df1.study.value_counts().index:
    print(study)
    print(len(df1[df1.study == study]))
    print(df1[df1.study == study].meaning_making.mean())
    print(df1[df1.study == study].meaning_making.value_counts())

# %%
df2 = pd.read_excel(start.DATA_DIR + "temp/negative_core_beliefs.xlsx")
# %%
for study in df2.study.value_counts().index:
    print(study)
    print(df2[df2.study == study].participant_id.nunique())
    print(len(df2[df2.study == study]))
    print(df2[df2.study == study].negative_belief_any.mean())
    print(df2[df2.study == study].negative_belief_any.value_counts())
# %%

long_df = pd.concat([df1, df2])
long_df[long_df.study == "recast"].participant_id.nunique()
long_df[long_df.study == "recast"].question.value_counts()
df1[df1.study == "recast"].question.value_counts()
df2[df2.study == "recast"].question.value_counts()

# %% Merge
recast_mm = df1[df1.study == "recast"]
recast_ncb = df2[df2.study == "recast"]

recast = recast_ncb[["participant_id", "question", "negative_belief_any"]].merge(
    recast_mm[["participant_id", "question", "meaning_making"]],
    on=["participant_id", "question"],
    how="outer",
    indicator=True,
)
recast = recast[recast._merge == "both"]

recast.question.value_counts()

recast[recast.question == "emotion"].negative_belief_any.value_counts()
recast[recast.question == "emotion"].negative_belief_any.mean()

recast[recast.question == "emotion"].meaning_making.value_counts()
recast[recast.question == "emotion"].meaning_making.mean()

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
