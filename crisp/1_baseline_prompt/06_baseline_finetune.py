import time
import os
from openai import OpenAI
from crisp.library import secrets, start
import pandas as pd

# ------------------ SETUP ------------------
client = OpenAI(api_key=secrets.OPENAI_API_KEY)

CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL

EXPORT_DIR = start.DATA_DIR + "finetune/"

###
# ------------------ Top Model ------------------
###
LABEL = "top"

FILENAME = f"{PLATFORM}_{CONCEPT}_baseline_finetune_train_{LABEL}.jsonl"
LOCAL_FILE_PATH = os.path.join(EXPORT_DIR, FILENAME)

print(f"Preparing to fine-tune {MODEL} for {CONCEPT} using {LABEL} prompt")
print(f"Loading file: {LOCAL_FILE_PATH}")

# ------------------ STEP 1: Upload File ------------------
with open(LOCAL_FILE_PATH, "rb") as f:
    upload_response = client.files.create(file=f, purpose="fine-tune")

top_file_id = upload_response.id
print(f"✅ Uploaded file. File ID: {top_file_id}")

# ------------------ STEP 2: Start Fine-Tune ------------------
top_fine_tune_response = client.fine_tuning.jobs.create(
    training_file=top_file_id,
    model=MODEL,
)

top_job_id = top_fine_tune_response.id
print(f"🚀 Started fine-tune job. Job ID: {top_job_id}")
print(f"📍 Status: {top_fine_tune_response.status}")

top_final_job = client.fine_tuning.jobs.retrieve(top_job_id)

if top_final_job.status == "succeeded":
    top_fine_tuned_model_name = top_final_job.fine_tuned_model
    print(f"✅ Fine-tuned model name: {top_fine_tuned_model_name}")

# top_fine_tuned_model_name = fine_tuned_model_name
else:
    print(f"❌ Fine-tune job failed with status: {top_final_job.status}")
    print(f"Error message: {top_final_job.error.message}")

###
# ------------------ Bottom Model ------------------
###
LABEL = "bottom"
FILENAME = f"{PLATFORM}_{CONCEPT}_baseline_finetune_train_{LABEL}.jsonl"
LOCAL_FILE_PATH = os.path.join(EXPORT_DIR, FILENAME)
print(f"Preparing to fine-tune {MODEL} for {CONCEPT} using {LABEL} prompt")
print(f"Loading file: {LOCAL_FILE_PATH}")
# ------------------ STEP 1: Upload File ------------------
with open(LOCAL_FILE_PATH, "rb") as f:
    upload_response = client.files.create(file=f, purpose="fine-tune")
bottom_file_id = upload_response.id
print(f"✅ Uploaded file. File ID: {bottom_file_id}")
# ------------------ STEP 2: Start Fine-Tune ------------------
bottom_fine_tune_response = client.fine_tuning.jobs.create(
    training_file=bottom_file_id,
    model=MODEL,
)
bottom_job_id = bottom_fine_tune_response.id
print(f"🚀 Started fine-tune job. Job ID: {bottom_job_id}")
print(f"📍 Status: {bottom_fine_tune_response.status}")  # HERE
bottom_final_job = client.fine_tuning.jobs.retrieve(bottom_job_id)
if bottom_final_job.status == "succeeded":
    bottom_fine_tuned_model_name = bottom_final_job.fine_tuned_model
    print(f"✅ Fine-tuned model name: {bottom_fine_tuned_model_name}")
else:
    print(f"❌ Fine-tune job failed with status: {bottom_final_job.status}")
    print(f"Error message: {bottom_final_job.error.message}")

# Export top and bottom fine-tuned model names to excel

export_df = pd.DataFrame(
    {
        "category": ["top", "bottom"],
        "fine_tuned_model_name": [
            top_fine_tuned_model_name,
            bottom_fine_tuned_model_name,
        ],
    }
)
export_df.to_excel(
    os.path.join(
        EXPORT_DIR, f"{PLATFORM}_{CONCEPT}_baseline_finetuned_model_names.xlsx"
    ),
    index=False,
)
