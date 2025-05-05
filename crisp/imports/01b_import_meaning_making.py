# %%
from crisp.library import start
import pandas as pd

# %%
df1a = pd.read_excel(
    start.RAW_DIR + "Meaning Making/ATTACHMENT MM.xlsx", sheet_name="MM = yes"
)
df1b = pd.read_excel(
    start.RAW_DIR + "Meaning Making/ATTACHMENT MM.xlsx", sheet_name="MM = no"
)

df1 = pd.concat([df1a, df1b])
df1 = df1.rename(
    columns={
        "id": "participant_id",
        "full quote": "text",
        "mm code": "meaning_making",
        "sentence 1": "meaning_making_sentence1",
        "SENTENCE 2": "meaning_making_sentence_2",
        "sentence 3": "meaning_making_sentence_3",
        "Unnamed: 6": "meaning_making_sentence_4",
        "sentence 5": "meaning_making_sentence_5",
        "Unnamed: 7": "meaning_making_sentence_6",
    }
)
# %%
# drop unnamed columns
df1 = df1.loc[:, ~df1.columns.str.contains("^Unnamed")]
df1.sample()
df1.meaning_making.value_counts()
df1.meaning_making.mean()
len(df1)
df1["question"] = "attachment"
df1["study"] = "recast"
# %%
df2a = pd.read_excel(
    start.RAW_DIR + "Meaning making/emotion climate MM.xlsx", sheet_name="MM = YES"
)
df2b = pd.read_excel(
    start.RAW_DIR + "Meaning making/emotion climate MM.xlsx", sheet_name="MM = No"
)

df2 = pd.concat([df2a, df2b])

df2 = df2.rename(
    columns={
        "prolificid": "participant_id",
        "How Effect Now": "text",
        "MMCODE": "meaning_making",
        "comments": "notes",
        "sentence 1": "meaning_making_sentence1",
        "sentence 2": "meaning_making_sentence_2",
        "Sentence 3": "meaning_making_sentence_3",
    }
)

df2 = df2[df2.meaning_making.isin([0, 1])]
df2.meaning_making.value_counts()
df2.meaning_making.mean()
len(df2)
df2["study"] = "recast"
df2["question"] = "emotion"

# %%


# %%
df3 = pd.read_excel(
    start.RAW_DIR + "Meaning Making/MM - MOTHER TURNING POINT.xlsx",
)
df3 = df3.rename(
    columns={
        "id": "participant_id",
        "quote": "text",
        "MX NEW CODE (0 - NO; 1=YES; 99 = CONSULT)": "meaning_making",
        "SENTENCE 1": "meaning_making_sentence1",
        "SENTENCE 2": "meaning_making_sentence_2",
        "SENTENCE 3": "meaning_making_sentence_3",
        "NOTES": "notes",
    }
)

df3.sample()
len(df3)
df3.meaning_making.value_counts()
df3["study"] = "mother"
df3["question"] = "turning point"

# %%
df4 = pd.read_excel(
    start.RAW_DIR + "Meaning Making/MM recast low point.xlsx",
)
df4 = df4.rename(
    columns={
        "prolificid": "participant_id",
        "lowpoint_desc": "text1",
        "locpoint_meaning": "text2",
        "meaning_low": "meaning_making",
        "sentence1": "meaning_making_sentence1",
        "sentence2": "meaning_making_sentence_2",
        "COMMENTS FOR DISCUSSION": "notes",
    }
)
df4["study"] = "recast"
df4["question"] = "low point"
len(df4)
df4.meaning_making.value_counts()
df4.meaning_making.mean()
df4.sample()
# TODO: Ask difference between 1 and 100, also text1 text2
# %%
df5 = pd.read_excel(
    start.RAW_DIR + "Meaning Making/sexual assault just meaning making.xlsx",
)

df5 = df5.rename(
    columns={
        "complete? 0 = cuffoff before event; 1= event but cutoff; 2 = seems complete": "cutoff",
        "id": "participant_id",
        "Meaning making (0=no; 1=yes)": "meaning_making",
        "SENTENCE": "meaning_making_sentence",
        "ISSUES FOR MM USE (1 = YES ISSUE THAT WOULD IMPACT TRAINING)": "coding_issue",
        "ISSUE": "coding_issue_type",
    }
)
df5.sample()
len(df5)
df5.meaning_making.value_counts()
df5["study"] = "sexual_assault"
df5["question"] = "unknown"
# %%
df = pd.concat([df1, df2, df3, df5])
# %%
df.meaning_making.value_counts()
# %%
df = df[df.meaning_making.isin([0, 1])]
df.coding_issue.value_counts()

df = df[df.coding_issue != 1]
df.study.value_counts()

# %%
# move question and study to the front after participant_id
cols = list(df.columns)
cols.remove("question")
cols.remove("study")
cols.remove("participant_id")
df = df[["participant_id", "question", "study"] + cols]
df.sample()

# %%

df.to_excel(start.DATA_DIR + "temp/meaning_making.xlsx", index=False)
# %%
