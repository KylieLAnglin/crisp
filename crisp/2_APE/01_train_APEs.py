# A3_ape_training.py
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import f1_score

from crisp.library import start, secrets
from crisp.library import format_prompts, classify

# ------------------ CONSTANTS ------------------

CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in train set")
SAMPLE = start.SAMPLE
NUM_VARIANTS = 5
NUM_GENERATIONS = 5

META_INSTRUCTIONS1 = "Generate a variation of the following instruction while keeping the output format. You can add important information or remove unnecessary information. Instruction:\n"
META_INSTRUCTIONS2 = "\nOutput only the new instruction."

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

BASELINE_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_results_dev.xlsx"
)


TRACKING_PATHS = {
    "top": {
        "csv": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_top_results.csv",
        "fig": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_evolution_top_train.png",
    },
    "bottom": {
        "csv": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_bottom_results.csv",
        "fig": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_evolution_bottom_train.png",
    },
}

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"].rename(columns={"human_code": "label"})
df = df[df.text.notna() & df.label.notna()]
if SAMPLE:
    df = df.sample(5, random_state=start.SEED)
prompt_df = pd.read_excel(BASELINE_RESULTS_PATH, sheet_name="results")


# ------------------ FUNCTIONS ------------------


def apply_prompt_to_classify(df, prompt):
    predictions = []
    for text in tqdm(df["text"], desc="Classifying texts"):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        response, _ = classify.format_message_and_get_response(
            model_provider=start.PLATFORM,
            prompt=messages,
            text_to_classify=text,
            temperature=0.0001,
        )
        prediction = classify.create_binary_classification_from_response(response)
        predictions.append(prediction)
    return predictions


def evaluate_prompt(df, prompt):
    preds = apply_prompt_to_classify(df, prompt)
    return f1_score(df["label"].tolist(), preds)


# ------------------ MAIN SCRIPT ------------------
prompts = [
    ("top", prompt_df["F1"].idxmax()),
    ("bottom", prompt_df["F1"].idxmin()),
]

for cat, index in prompts:
    print(f"\n======== Evaluating {cat.upper()} Prompt Seed ========")
    OUTPUT_TRACKING_FILE = TRACKING_PATHS[cat]["csv"]
    FIGURE_FILE = TRACKING_PATHS[cat]["fig"]

    STARTING_PROMPT = prompt_df.loc[index, "Prompt"]
    tracking_records = []
    current_prompt = STARTING_PROMPT

    for generation in range(NUM_GENERATIONS):
        print(f"\n=== Generation {generation+1} ===")
        variants = classify.generate_prompt_variants(
            model_provider=start.PLATFORM,
            base_prompt=current_prompt,
            metaprompt1=META_INSTRUCTIONS1,
            metaprompt2=META_INSTRUCTIONS2,
            num_variants=NUM_VARIANTS,
        )
        variant_scores = []
        for idx, variant in enumerate(variants):
            print(f"Evaluating variant {idx+1}...")
            f1 = evaluate_prompt(df, variant)
            variant_scores.append((variant, f1))
            print(f"Variant F1: {f1:.4f}")
            tracking_records.append(
                {
                    "generation": generation + 1,
                    "variant_id": idx + 1,
                    "prompt": variant,
                    "f1_score": f1,
                }
            )

        best_variant, best_f1 = max(variant_scores, key=lambda x: x[1])
        print(f"Best variant selected with F1 {best_f1:.4f}")
        current_prompt = best_variant

    tracking_df = pd.DataFrame(tracking_records)
    tracking_df.to_csv(OUTPUT_TRACKING_FILE, index=False)

# ------------------ PLOT RESULTS ------------------
for mode, _ in prompts:
    tracking_df = pd.read_csv(TRACKING_PATHS[mode]["csv"])
    tracking_df["generation"] = tracking_df["generation"].astype(int)
    tracking_df["f1_score"] = tracking_df["f1_score"].astype(float)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        tracking_df["generation"], tracking_df["f1_score"], color="black", alpha=0.7
    )
    plt.plot(
        tracking_df.groupby("generation")["f1_score"].max(),
        marker="o",
        linestyle="-",
        color="black",
    )

    plt.xlabel("Generation")
    plt.ylabel("F1 Score")
    plt.title(f"Prompt F1 Scores Across Generations - {mode.title()} Seed")
    plt.grid(True, linestyle="--", color="gray", alpha=0.7)
    plt.tight_layout()
    plt.savefig(TRACKING_PATHS[mode]["fig"])
    plt.show()
# %%
