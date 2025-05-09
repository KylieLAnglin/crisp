# %%
from openai import OpenAI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score

from crisp.library import start
from crisp.library import secrets

# ------------------ CONSTANTS ------------------
NUM_VARIANTS = 5
NUM_GENERATIONS = 5
MODEL = start.MODEL
META_INSTRUCTIONS1 = "Generate a variation of the following instruction while keeping the output format. You can add important information or remove unnecessary information. Instruction:\n"
META_INSTRUCTIONS2 = "\nOutput only the new instruction."

OPENAI_API_KEY = secrets.OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_FILE = "ncb_variants"
PROMPT_PATH = start.MAIN_DIR + "results/" + PROMPT_FILE + "_results_dev.xlsx"
prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="combo_results")

# %%
# ------------------ IMPORTS ------------------
df = pd.read_excel(start.DATA_DIR + "clean/negative_core_beliefs.xlsx")
df = df[df.split_group == "train"]
df = df.rename(columns={"human_code": "label"})
df = df[df.text.notna()]
df = df[df.label.notna()]


# %%
# ------------------ FUNCTIONS ------------------
def generate_prompt_variants(base_prompt, num_variants):
    variants = []
    for _ in range(num_variants):
        meta_instructions = META_INSTRUCTIONS1 + base_prompt + META_INSTRUCTIONS2
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": meta_instructions}],
            temperature=1,
        )
        new_prompt = response.choices[0].message.content.strip()
        variants.append(new_prompt)
    return variants


def apply_prompt_to_classify(df, prompt):
    predictions = []
    for text in tqdm(df["text"], desc="Classifying texts"):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.00001,
            n=1,
        )
        prediction = parse_prediction(response.choices[0].message.content)
        predictions.append(prediction)
    return predictions


def parse_prediction(response_text):
    response_text = response_text.strip().lower()
    if "yes" in response_text:
        return 1
    elif "no" in response_text:
        return 0
    else:
        return 0


def evaluate_prompt(df, prompt):
    preds = apply_prompt_to_classify(df, prompt)
    true_labels = df["label"].tolist()
    return f1_score(true_labels, preds)


# %%
# ------------------ SCRIPT BLOCK ------------------
cases = [
    (
        "best",
        prompt_df["F1"].idxmax(),
        "ncb_ape_best",
        "prompt_evolution_tracking_best.png",
    ),
    (
        "worst",
        prompt_df["F1"].idxmin(),
        "ncb_ape_worst",
        "prompt_evolution_tracking_worst.png",
    ),
]

for mode, index, csv_name, fig_name in cases:
    print(f"\n======== Evaluating {mode.upper()} Prompt Seed ========")
    OUTPUT_TRACKING_FILE = start.DATA_DIR + f"temp/{csv_name}.csv"
    FIGURE_FILE = start.RESULTS_DIR + fig_name

    STARTING_PROMPT = prompt_df.loc[index, "Prompt"]
    tracking_records = []
    current_prompt = STARTING_PROMPT

    for generation in range(NUM_GENERATIONS):
        print(f"\n=== Generation {generation+1} ===")
        variants = generate_prompt_variants(current_prompt, NUM_VARIANTS)
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

    print("\nFinal optimized prompt:\n")
    print(current_prompt)
    print(f"\nTracking CSV saved to: {OUTPUT_TRACKING_FILE}")

    print("\nTop 5 Prompts by F1 Score:\n")
    top5 = tracking_df.sort_values(by="f1_score", ascending=False).head(5)
    for idx, row in top5.iterrows():
        print(
            f"F1: {row['f1_score']:.4f} | Generation {row['generation']} Variant {row['variant_id']}\nPrompt: {row['prompt']}\n"
        )

# %%
for mode, index, csv_name, fig_name in cases:
    OUTPUT_TRACKING_FILE = start.DATA_DIR + f"temp/{csv_name}.csv"
    FIGURE_FILE = start.RESULTS_DIR + fig_name

    # Load tracking data
    tracking_df = pd.read_csv(OUTPUT_TRACKING_FILE)
    tracking_df["generation"] = tracking_df["generation"].astype(int)

    # plot F1 scores across generations
    tracking_df["f1_score"] = tracking_df["f1_score"].astype(float)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(
        tracking_df["generation"],
        tracking_df["f1_score"],
        color="black",
        alpha=0.7,
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
    plt.savefig(FIGURE_FILE)
    plt.show()
# %%
