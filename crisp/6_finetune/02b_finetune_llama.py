# llama3_finetune_train.py
# %%
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from crisp.library import start

# ------------------ CONSTANTS ------------------
CONCEPT = start.CONCEPT
PLATFORM = "llama"
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED
LABEL = "top"  # change to "bottom" as needed
TECHNIQUE = "baseline"

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} using {TECHNIQUE}-{LABEL}")

DATA_PATH = (
    start.DATA_DIR
    + f"finetune_llama/{PLATFORM}_{CONCEPT}_{TECHNIQUE}_finetune_train_{LABEL}.jsonl"
)
OUTPUT_DIR = start.RESULTS_DIR + f"llama3_{CONCEPT}_{TECHNIQUE}_{LABEL}_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD MODEL AND TOKENIZER ------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
)

# ------------------ APPLY LoRA ------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)


# ------------------ LOAD AND FORMAT DATA ------------------
def format_instruction(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
    }


raw_dataset = load_dataset("json", data_files=DATA_PATH)["train"]
if SAMPLE:
    raw_dataset = raw_dataset.shuffle(seed=SEED).select(range(5))

formatted_dataset = raw_dataset.map(format_instruction)
tokenized_dataset = formatted_dataset.map(
    lambda x: tokenizer(
        x["text"], padding="max_length", truncation=True, max_length=512
    ),
    batched=True,
    remove_columns=formatted_dataset.column_names,
)

# ------------------ TRAINING ARGS ------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
)

# ------------------ TRAIN ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
# %%
