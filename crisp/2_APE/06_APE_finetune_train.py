import time
import os
import pandas as pd
from openai import OpenAI
from crisp.library import secrets, start

# ------------------ SETUP ------------------
client = OpenAI(api_key=secrets.OPENAI_API_KEY)

CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
EXPORT_DIR = start.DATA_DIR + "finetune/"
SUFFIX_TEMPLATE = f"{PLATFORM}_{CONCEPT}_APE_{{label}}"

# ------------------ File Setup ------------------
labels = ["top", "bottom"]
file_ids = {}
job_ids = {}
final_model_names = {}

# ------------------ STEP 0: Get list of existing fine-tune jobs ------------------
existing_suffixes = set()
all_jobs = client.fine_tuning.jobs.list(limit=100)

for job_summary in all_jobs:
    model_name = job_summary.fine_tuned_model
    if model_name is not None:
        # print(f"Checking model: {model_name}")
        if CONCEPT in model_name:
            if PLATFORM in model_name:
                if "ape" in model_name:
                    suffix = model_name.split(SUFFIX_TEMPLATE)[-1]
                    existing_suffixes.add(suffix)
                    print(f"Found existing model with suffix: {suffix}")

# ------------------ STEP 1: Upload and fine-tune if not already done ------------------
for label in labels:
    for suffix in existing_suffixes:
        if label in suffix:
            print(f"⚠️ Skipping {label} — model with suffix '{suffix}' already exists.")
            SKIP = True
    if SKIP == False:
        # Upload file
        filename = f"{PLATFORM}_{CONCEPT}_APE_finetune_train_{label}.jsonl"
        local_file_path = os.path.join(EXPORT_DIR, filename)
        print(f"📤 Uploading file for {label} prompt: {local_file_path}")

        with open(local_file_path, "rb") as f:
            upload_response = client.files.create(file=f, purpose="fine-tune")

        file_ids[label] = upload_response.id
        print(f"✅ Uploaded {label} file. File ID: {file_ids[label]}")

        # Start fine-tune job
        print(f"🚀 Starting fine-tune for {label} with suffix '{suffix}'...")
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=file_ids[label],
            model=MODEL,
            suffix=suffix,
        )
        job_ids[label] = fine_tune_response.id
        print(f"🆔 Job ID for {label}: {job_ids[label]}")
        print(f"📍 Status: {fine_tune_response.status}")

# ------------------ STEP 2: Retrieve job status and model names ------------------
for label in labels:
    if label not in job_ids:
        print(f"⏭️ Skipping model retrieval for {label} — no job was started.")
        continue

    print(f"🔍 Checking final status for {label} model...")
    final_job = client.fine_tuning.jobs.retrieve(job_ids[label])

    if final_job.status == "succeeded":
        final_model_names[label] = final_job.fine_tuned_model
        print(f"✅ Fine-tuned model ({label}): {final_model_names[label]}")
    else:
        final_model_names[label] = None
        print(f"❌ Fine-tune job for {label} failed. Status: {final_job.status}")
        if hasattr(final_job, "error"):
            print(f"Error message: {final_job.error.message}")

# ------------------ STEP 3: Export model names to Excel ------------------
export_df = pd.DataFrame(
    {
        "category": ["top", "bottom"],
        "fine_tuned_model_name": [
            final_model_names.get("top"),
            final_model_names.get("bottom"),
        ],
    }
)
export_path = os.path.join(
    EXPORT_DIR, f"{PLATFORM}_{CONCEPT}_APE_finetuned_model_names.xlsx"
)
export_df.to_excel(export_path, index=False)
print(f"\n📁 Exported model names to {export_path}")
