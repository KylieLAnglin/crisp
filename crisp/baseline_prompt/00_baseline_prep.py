# %%
import pandas as pd
from crisp.library import start

# ------------------ CONSTANTS ------------------
PROMPT_FILE = "ncb_baseline_variants"
source_path = start.DATA_DIR + f"prompts/{PROMPT_FILE}.xlsx"
output_sheet = "variants"

# ------------------ LOAD AND COMBINE PROMPTS ------------------
part1_variants = pd.read_excel(source_path, sheet_name="part1", index_col="part_num")
part2_variants = pd.read_excel(source_path, sheet_name="part2", index_col="part_num")

prompt_combos = [
    {
        "prompt": f"{part1} {part2}",
        "part1": part1_id,
        "part2": part2_id,
    }
    for part1_id, part1 in part1_variants["prompt_part"].items()
    for part2_id, part2 in part2_variants["prompt_part"].items()
]

combo_df = pd.DataFrame(prompt_combos)
combo_df["baseline_prompt_id"] = combo_df.index

with pd.ExcelWriter(
    source_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
) as writer:
    combo_df.to_excel(writer, sheet_name=output_sheet, index=False)

print(
    f"Exported {len(combo_df)} prompt combinations to sheet '{output_sheet}' in {source_path}"
)
# %%
