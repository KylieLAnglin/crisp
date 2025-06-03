# %%
from crisp.library import start
import pandas as pd

# %%
recast_identity = pd.read_excel(
    start.RAW_DIR + "Negative core beliefs/recast_identity task.xlsx"
)
recast_identity_columns = {
    "id": "participant_id",
    "response": "text",
    "self": "negative_belief_self",
    "other": "negative_belief_others",
    "trust": "negative_belief_trust",
    "NOTES ON CODING ISSUES": "coding_issue_type",
}
recast_identity = recast_identity.rename(columns=recast_identity_columns)
recast_identity = recast_identity[recast_identity_columns.values()]

recast_identity["negative_belief_any"] = recast_identity[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_identity.negative_belief_any = recast_identity.negative_belief_any.replace(
    99, pd.NA
)
recast_identity["study"] = "recast"
recast_identity["question"] = "identity"


# %%
recast_emotion = pd.read_excel(
    start.RAW_DIR + "Negative core beliefs/recast_emotion socialization.xlsx"
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
recast_emotion = recast_emotion.rename(columns=recast_emotion_columns)
recast_emotion = recast_emotion[recast_emotion_columns.values()]

recast_emotion["negative_belief_any"] = recast_emotion[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_emotion.negative_belief_any = recast_emotion.negative_belief_any.replace(
    99, pd.NA
)
recast_emotion["study"] = "recast"
recast_emotion["question"] = "emotion"
# %%

recast_attachment = pd.read_excel(
    start.RAW_DIR + "Negative core beliefs/recast_attachment task.xlsx"
)
recast_attachment_columns = {
    "id": "participant_id",
    "how impact now": "text",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --SELF": "negative_belief_self",
    "COPY AND PASTE NEGATIVE CORE BELIEF SENTENCE": "negative_belief_self_sentence",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --others/world": "negative_belief_others",
    "COPY AND PASTE NEGATIVE CORE BELIEF SENTENCE.1": "negative_belief_others_sentence",
    "INCLUDES NEGATIVE CORE/GENERALIZED BELIEF --TRUST": "negative_belief_trust",
    "COPY AND PASTE NEGATIVE CORE BELIEF SENTENCE.2": "negative_belief_trust_sentence",
    "ISSUE FOR TRAINING (1 = YES, but can use; 2 = don't use b/c unresolved coding)": "coding_issue",
    "TYPE OF ISSUE": "coding_issue_type",
}

recast_attachment = recast_attachment.rename(columns=recast_attachment_columns)
recast_attachment = recast_attachment[recast_attachment_columns.values()]

recast_attachment["negative_belief_any"] = recast_attachment[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
recast_attachment.negative_belief_any = recast_attachment.negative_belief_any.replace(
    99, pd.NA
)
recast_attachment["study"] = "recast"
recast_attachment["question"] = "attachment"
# %%

# %%
df = pd.concat([recast_identity, recast_emotion, recast_attachment], ignore_index=True)

# %%
df["negative_belief_any"] = df[
    ["negative_belief_self", "negative_belief_others", "negative_belief_trust"]
].max(axis=1)
df.negative_belief_any = df.negative_belief_any.replace(99, pd.NA)
# %%
df.study.value_counts()

# %%
df.to_excel(start.DATA_DIR + "temp/negative_core_beliefs.xlsx", index=False)

