# %%
from crisp.library import start
import pandas as pd

# %%

recast_emotion_rater1 = pd.read_excel(
    start.RAW_DIR
    + "Negative core beliefs/neg beliefs raters coding/recast_emotion socialization_WITH RATER DATA.xlsx",
    sheet_name="rater1 grad 50",
)
recast_emotion_rater2 = pd.read_excel(
    start.RAW_DIR
    + "Negative core beliefs/neg beliefs raters coding/recast_emotion socialization_WITH RATER DATA.xlsx",
    sheet_name="rater 2 grad 50",
)

recast_emotion_columns = {
    "prolificid": "participant_id",
    "How Effect Now": "text",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --SELF (0 = no; 1= yes)": "negative_belief_self",
    "COPY AND PASTE NEGATIVE CORE BELIEF SENTENCE": "negative_belief_self_sentence",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --others/world (0 = no' 1 = yes)": "negative_belief_others",
    "COPY AND PASTE NEGATIVE CORE BELIEF SENTENCE.1": "negative_belief_others_sentence",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --TRUST": "negative_belief_trust",
    "COPY AND PASTE NEGATIVE CORE BELIEF SENTENCE.2": "negative_belief_trust_sentence",
    "CONTENT QUESTION FOR REVIEW": "coding_issue",
    "NOTES": "coding_issue_type",
}
recast_emotion_rater1 = recast_emotion_rater1.rename(columns=recast_emotion_columns)
recast_emotion_rater1 = recast_emotion_rater1[recast_emotion_columns.values()]

recast_emotion_rater1["rater1_code"] = recast_emotion_rater1[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_emotion_rater1.rater1_code = recast_emotion_rater1.rater1_code.replace(99, pd.NA)
recast_emotion_rater1 = recast_emotion_rater1[["participant_id", "rater1_code"]]

recast_emotion_rater2 = recast_emotion_rater2.rename(columns=recast_emotion_columns)
recast_emotion_rater2 = recast_emotion_rater2[recast_emotion_columns.values()]
recast_emotion_rater2["rater2_code"] = recast_emotion_rater2[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_emotion_rater2.rater2_code = recast_emotion_rater2.rater2_code.replace(99, pd.NA)
recast_emotion_rater2 = recast_emotion_rater2[["participant_id", "rater2_code"]]

recast_emotion = recast_emotion_rater1.merge(
    recast_emotion_rater2,
    on=["participant_id"],
    how="outer",
    indicator=True,
)

# delete where rater1_code or rater2_code is missing
recast_emotion = recast_emotion[
    (recast_emotion.rater1_code.notna()) & (recast_emotion.rater2_code.notna())
]
recast_emotion["study"] = "recast"
recast_emotion["question"] = "emotion"
recast_emotion["construct"] = "ncb"

# %%
recast_attachment_rater1 = pd.read_excel(
    start.RAW_DIR
    + "Negative core beliefs/neg beliefs raters coding/recast attach neg beliefs raters.xlsx",
    sheet_name="rater 1 gradx",
)

recast_attachment_rater2 = pd.read_excel(
    start.RAW_DIR
    + "Negative core beliefs/neg beliefs raters coding/recast attach neg beliefs raters.xlsx",
    sheet_name="rater 2 kc",
)
# recast_emotion.to_excel(start.DATA_DIR + "clean/recast_emotion_irr.xlsx", index=False)
recast_attachment_columns = {
    "id": "participant_id",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --SELF": "negative_belief_self",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --others/world": "negative_belief_others",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --TRUST": "negative_belief_trust",
}
recast_attachment_rater1 = recast_attachment_rater1.rename(
    columns=recast_attachment_columns
)
recast_attachment_rater1 = recast_attachment_rater1[recast_attachment_columns.values()]
recast_attachment_rater1["rater1_code"] = recast_attachment_rater1[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_attachment_rater1.rater1_code = recast_attachment_rater1.rater1_code.replace(
    99, pd.NA
)
recast_attachment_rater1 = recast_attachment_rater1[["participant_id", "rater1_code"]]

recast_attachment_rater2 = recast_attachment_rater2.rename(
    columns=recast_attachment_columns
)
recast_attachment_rater2 = recast_attachment_rater2[recast_attachment_columns.values()]
recast_attachment_rater2["rater2_code"] = recast_attachment_rater2[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_attachment_rater2.rater2_code = recast_attachment_rater2.rater2_code.replace(
    99, pd.NA
)
recast_attachment_rater2 = recast_attachment_rater2[["participant_id", "rater2_code"]]


recast_attachment = recast_attachment_rater1.merge(
    recast_attachment_rater2,
    on=["participant_id"],
    how="outer",
    indicator=True,
)
# %%

ncb_df = pd.concat(
    [recast_emotion, recast_attachment], ignore_index=True
)  # , ignore_index=True)
ncb_df = ncb_df[
    [
        "participant_id",
        "study",
        "question",
        "construct",
        "rater1_code",
        "rater2_code",
    ]
]
ncb_df.to_excel(start.DATA_DIR + "clean/ncb_irr.xlsx", index=False)
