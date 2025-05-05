# %%
from library import start
import pandas as pd

# %%
attachment_df1 = pd.read_excel(
    start.RAW_DIR + "Meaning Making/ATTACHMENT MM.xlsx", sheet_name="MM = yes"
)

attachment_df2 = pd.read_excel(
    start.RAW_DIR + "Meaning Making/ATTACHMENT MM.xlsx", sheet_name="MM = no"
)

attachment_df = pd.concat([attachment_df1, attachment_df2])
attachment_df = attachment_df.rename(
    columns={
        "full quote": "writing",
        "mm code": "mm_code",
        "sentence 1": "sentence1",
        "SENTENCE 2": "sentence_2",
        "sentence 3": "sentence_3",
        "Unnamed: 6": "sentence_4",
        "sentence 5": "sentence_5",
        "Unnamed: 7": "sentence_6",
    }
)
attachment_df.sample()
attachment_df.mm_code.value_counts()
# %%
attachment_df.to_excel(start.DATA_DIR + "clean/attachment.xlsx", index=False)

# %%
