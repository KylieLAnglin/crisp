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
MODEL = "gpt-4.1"
STARTING_PROMPT = "Negative core beliefs refer to broad, generalized, or exaggerated judgments people have about themselves, others, or the world. Negative core beliefs are often inaccurate, harmful, or just not useful. The key idea is that negative core beliefs are more generalized than warranted. Does the following text contain an example of the psychological concept of a negative core belief? Respond Yes or No. Text: "
META_INSTRUCTIONS1 = "Generate a variation of the following instruction while keeping the output format. You can add important information or remove unnecessary information. Instruction:\n"
META_INSTRUCTIONS2 = "\nOutput only the new instruction."
OUTPUT_TRACKING_FILE = (
    start.RESULTS_DIR + "prompt_evolution_tracking_empirical_prompt.csv"
)

OPENAI_API_KEY = secrets.OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------ IMPORTS ------------------
# Assume df is loaded with 'text' and 'label' columns
df = pd.read_csv(start.DATA_DIR + "clean/negative_core_beliefs_train_gold.csv")
# remove missing text
df = df[df.text.notna()]
# remove missing label
df = df[df.label.notna()]

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
            temperature=0,
        )
        prediction = parse_prediction(response.choices[0].message.content)
        predictions.append(prediction)
    return predictions


def parse_prediction(response_text):
    """
    Modify this function based on how you want to parse the model's output.
    Assume binary classification: returns 0 or 1.
    """
    response_text = response_text.strip().lower()
    if "yes" in response_text:
        return 1
    elif "no" in response_text:
        return 0
    else:
        # Fallback in case of unclear response
        return 0


def evaluate_prompt(df, prompt):
    preds = apply_prompt_to_classify(df, prompt)
    true_labels = df["label"].tolist()
    return f1_score(true_labels, preds)


# %%
# ------------------ MAIN SCRIPT ------------------


# Initialize tracking dataframe
tracking_records = []

# Initialize with starting prompt
current_prompt = STARTING_PROMPT

for generation in range(NUM_GENERATIONS):
    print(f"\n=== Generation {generation+1} ===")

    # 1. Generate variants
    variants = generate_prompt_variants(current_prompt, NUM_VARIANTS)

    # 2. Evaluate variants
    variant_scores = []
    for idx, variant in enumerate(variants):
        print(f"Evaluating variant {idx+1}...")
        f1 = evaluate_prompt(df, variant)
        variant_scores.append((variant, f1))
        print(f"Variant F1: {f1:.4f}")

        # Save tracking info
        tracking_records.append(
            {
                "generation": generation + 1,
                "variant_id": idx + 1,
                "prompt": variant,
                "f1_score": f1,
            }
        )

    # 3. Select best variant
    best_variant, best_f1 = max(variant_scores, key=lambda x: x[1])
    print(f"Best variant selected with F1 {best_f1:.4f}")

    # 4. Update current prompt
    current_prompt = best_variant

# Save final tracking records
tracking_df = pd.DataFrame(tracking_records)
tracking_df.to_csv(OUTPUT_TRACKING_FILE, index=False)

print("\nFinal optimized prompt:\n")
print(current_prompt)
print(f"\nTracking CSV saved to: {OUTPUT_TRACKING_FILE}")

# Print top 5 prompts overall
print("\nTop 5 Prompts by F1 Score:\n")
top5 = tracking_df.sort_values(by="f1_score", ascending=False).head(5)
for idx, row in top5.iterrows():
    print(
        f"F1: {row['f1_score']:.4f} | Generation {row['generation']} Variant {row['variant_id']}\nPrompt: {row['prompt']}\n"
    )

# Plot F1 scores across generations
# Plot F1 scores across generations with jitter
plt.figure(figsize=(10, 6))
for generation, group in tracking_df.groupby("generation"):
    x_positions = np.random.normal(
        loc=generation, scale=0.1, size=len(group)
    )  # add jitter
    plt.scatter(
        x_positions, group["f1_score"], label=f"Generation {generation}", alpha=0.7
    )

# Plot best F1 per generation
plt.plot(
    tracking_df.groupby("generation")["f1_score"].max(),
    marker="o",
    linestyle="-",
    color="black",
    label="Best F1 per Generation",
)

plt.xlabel("Generation")
plt.ylabel("F1 Score")
plt.title("Prompt F1 Scores Across Generations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(start.RESULTS_DIR + "prompt_evolution_tracking_empirical_prompt.png")
