# %%
import pandas as pd
import numpy as np
from crisp.library import start

np.random.seed(123)
# ------------------ CONSTANTS ------------------
PROMPT_FILE = "ncb_baseline_variants"
source_path = start.DATA_DIR + f"prompts/{PROMPT_FILE}.xlsx"
output_sheet = "variants"

# ------------------ LOAD AND COMBINE PROMPTS ------------------
part1_variants = pd.read_excel(source_path, sheet_name="part1", index_col="part_num")
part2_variants = pd.read_excel(source_path, sheet_name="part2", index_col="part_num")
opt_variants = pd.read_excel(source_path, sheet_name="opt", index_col="opt_num")

sample_prompts = []
for sample_num in range(1, 51):
    # random sample from part 1
    part1_sample = part1_variants.sample(1, random_state=sample_num).iloc[0].iloc[0]
    # random sample number of additions - 0 through 3
    num_additions = np.random.randint(0, 4)
    # if num_additions > 0, random sample from opt without replacement
    if num_additions > 0:
        opt_sample = opt_variants.sample(num_additions, random_state=sample_num).iloc[
            :, 0
        ]
        opt_text = ""
        for opt in opt_sample:
            opt_text += f"{opt} "
    else:
        opt_text = ""
    # random sample from part 2
    part2_sample = part2_variants.sample(1, random_state=sample_num).iloc[0].iloc[0]
    # combine samples, add " Text: " to end
    sample_prompt = f"{part1_sample} {opt_text}{part2_sample} "
    print(sample_prompt)
    sample_prompts.append(sample_prompt)

# %%
prompt_df = pd.DataFrame(sample_prompts, columns=["prompt"])
prompt_df["baseline_prompt_id"] = prompt_df.index + 1

prompt_df = prompt_df[["baseline_prompt_id", "prompt"]]


# %%

with pd.ExcelWriter(
    source_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
) as writer:
    prompt_df.to_excel(writer, sheet_name=output_sheet, index=False)


# %%
