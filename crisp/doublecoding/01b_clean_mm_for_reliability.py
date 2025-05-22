# %%
from crisp.library import start
import pandas as pd

# %%
recast_emotion = pd.read_excel(
    start.RAW_DIR
    + "Meaning making/meaning making rater files/RECAST EMOTION RATING.xlsx"
)
recast_emotion_columns = {
    "prolificid": "participant_id",
    "MM_UG1_AR": "rater1_code",
    "MM_UG2_AS": "rater2_code",
    "MM_UG3_VB": "rater3_code",
}
recast_emotion = recast_emotion.rename(columns=recast_emotion_columns)
recast_emotion = recast_emotion[recast_emotion_columns.values()]
for rater in ["rater1_code", "rater2_code", "rater3_code"]:
    recast_emotion[rater] = recast_emotion[rater].replace(99, pd.NA)

recast_emotion["num_raters"] = (
    recast_emotion[["rater1_code", "rater2_code", "rater3_code"]].notna().sum(axis=1)
)

recast_emotion = recast_emotion[recast_emotion.num_raters > 1]
recast_emotion["study"] = "recast"
recast_emotion["question"] = "emotion"
# %%
recast_attachment = pd.read_excel(
    start.RAW_DIR
    + "Meaning making/meaning making rater files/recast_attachment_MMrater.xlsx",
    sheet_name="MM_attachment_rater data",
)

recast_attachment_columns = {
    "prolificid": "participant_id",
    "MEANING_ug1": "rater1_code",
    "MEANING_ug2": "rater2_code",
    "MEANING_ug3": "rater3_code",
    "meaning_grad": "rater4_code",
}
recast_attachment = recast_attachment.rename(columns=recast_attachment_columns)
recast_attachment = recast_attachment[recast_attachment_columns.values()]
for rater in ["rater1_code", "rater2_code", "rater3_code", "rater4_code"]:
    recast_attachment[rater] = recast_attachment[rater].replace(99, pd.NA)
    recast_attachment[rater] = recast_attachment[rater].replace("<NA>", pd.NA)
recast_attachment["num_raters"] = (
    recast_attachment[["rater1_code", "rater2_code", "rater3_code", "rater4_code"]]
    .notna()
    .sum(axis=1)
)
recast_attachment = recast_attachment[recast_attachment.num_raters > 1]
recast_attachment["study"] = "recast"
recast_attachment["question"] = "attachment"
# %%
mother_df = pd.read_excel(
    start.RAW_DIR
    + "Meaning making/meaning making rater files/mothermh_meaning_raters.xlsx",
    sheet_name="mothermh_meaning_raters",
)
mother_df_columns = {
    "prolificid": "participant_id",
    "UG_rater1": "rater1_code",
    "UG_rater2": "rater2_code",
}
mother_df = mother_df.rename(columns=mother_df_columns)
mother_df = mother_df[mother_df_columns.values()]
for rater in ["rater1_code", "rater2_code"]:
    mother_df[rater] = mother_df[rater].replace(99, pd.NA)
    mother_df[rater] = mother_df[rater].replace("<NA>", pd.NA)
mother_df["num_raters"] = mother_df[["rater1_code", "rater2_code"]].notna().sum(axis=1)
mother_df = mother_df[mother_df.num_raters > 1]
mother_df = mother_df[["participant_id", "rater1_code", "rater2_code"]]

mother_df["study"] = "mother"
mother_df["question"] = "turning point"

mm_df = pd.concat([recast_emotion, recast_attachment, mother_df])
mm_df = mm_df[
    [
        "participant_id",
        "study",
        "question",
        "rater1_code",
        "rater2_code",
        "rater3_code",
        "rater4_code",
        "num_raters",
    ]
]
mm_df["construct"] = "mm"  # meaning making
mm_df.to_excel(start.DATA_DIR + "clean/mm_irr.xlsx", index=False)
