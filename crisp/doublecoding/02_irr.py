# %%
# %%
from crisp.library import start
import pandas as pd
import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score

# %%
ncb_df = pd.read_excel(start.DATA_DIR + "clean/ncb_irr.xlsx")
mm_df = pd.read_excel(start.DATA_DIR + "clean/mm_irr.xlsx")


# %%
def get_irr(df, rater_columns):
    ratings = df[rater_columns]
    data_matrix = ratings.T.values
    data_matrix = pd.DataFrame(data_matrix).values
    alpha = krippendorff.alpha(
        reliability_data=data_matrix, level_of_measurement="nominal"
    )
    return alpha


ncb_alpha = get_irr(ncb_df, ["rater1_code", "rater2_code"])
mm_alpha = get_irr(mm_df, ["rater3_code", "rater4_code"])

print(f"NCB Krippendorff's alpha: {ncb_alpha}")
print(f"MM Krippendorff's alpha: {mm_alpha}")


# Krippendorff 2004


# %%
def get_kappa(df, col1, col2):
    # Drop rows with missing values in either column
    df_clean = df[[col1, col2]].dropna()
    return cohen_kappa_score(df_clean[col1], df_clean[col2])


ncb_kappa = get_kappa(ncb_df, "rater1_code", "rater2_code")
mm_kappa = get_kappa(mm_df, "rater3_code", "rater4_code")

print(f"NCB Cohen's kappa: {ncb_kappa}")
print(f"MM Cohen's kappa: {mm_kappa}")
# %%
