# %%
import pandas as pd

from library import start

# %%

df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df[df.split_group == "train"].human_code.value_counts()
