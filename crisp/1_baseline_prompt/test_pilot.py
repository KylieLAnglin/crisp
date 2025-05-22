# 01_baseline_train.py
# %%
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from crisp.library import start, secrets

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = "gemma"
MODEL = "google/gemma-2b"
SAMPLE = True
SEED = start.SEED

# Authenticate with OpenAI and HuggingFace
login(token="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")

print(
    f"Running baseline classification for '{CONCEPT}' on {PLATFORM} with {MODEL} (train set)"
)

# Set random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------ LOAD GEMMA MODEL ------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,  # Use float32 for CPU inference
    device_map={"": "cpu"},
)
model.eval()

# ------------------ PATHS ------------------
PROMPT_PATH = os.path.join(start.DATA_DIR, f"prompts/{CONCEPT}_baseline_variants.xlsx")
DATA_PATH = os.path.join(start.DATA_DIR, f"clean/{CONCEPT}.xlsx")
RESPONSE_PATH = os.path.join(
    start.DATA_DIR, f"responses_train/{PLATFORM}_{CONCEPT}_pilot.xlsx"
)
RESULTS_PATH = os.path.join(start.MAIN_DIR, f"results/{PLATFORM}_{CONCEPT}_pilot.xlsx")

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD BASELINE PROMPT ------------------
prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="baseline").head(1)
PROMPT = (
    "Meaning making is the process of restoring meaning after a highly stressful situation. "
    "Some examples of phrases associated with meaning making include: led me to be more accepting of things, "
    "taught me how to adjust to things I cannot change, taught me to be patient, brought my family closer together, "
    "helped me become a stronger person, helped me to grow emotionally and spiritually, made me more compassionate to those "
    "in similar situations, taught me that everyone has a right to be valued, led me to place less emphasis on material things, "
    "led me to change my priorities in life. Meaning making is associated with better mental and physical health. "
    "Does the text contain evidence of meaning making? Respond Yes or No."
)

# ------------------ GENERATE RESPONSES ------------------
completions = []

for text in tqdm(df.text, desc="Classifying"):
    full_prompt = f"Text to classify: {text}\n\nTask:\n{PROMPT}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = decoded[len(full_prompt) :].strip()
    completions.append(completion)

# TODO: Add response saving, evaluation, and results export if needed
