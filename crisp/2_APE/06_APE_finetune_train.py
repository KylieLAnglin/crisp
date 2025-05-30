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
# ------------------ File Setup ------------------
labels = ["top", "bottom"]
file_ids = {}
job_ids = {}
final_model_names = {}

# ------------------ STEP 1: Upload both files ------------------
for label in labels:
    filename = f"{PLATFORM}_{CONCEPT}_APE_finetune_train_{label}.jsonl"
    local_file_path = os.path.join(EXPORT_DIR, filename)

    print(f"📤 Uploading file for {label} prompt: {local_file_path}")
    with open(local_file_path, "rb") as f:
        upload_response = client.files.create(file=f, purpose="fine-tune")

    file_ids[label] = upload_response.id
    print(f"✅ Uploaded {label} file. File ID: {file_ids[label]}")

# ------------------ STEP 2: Start both fine-tune jobs ------------------
for label in labels:
    print(f"🚀 Starting fine-tune for {label}...")
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=file_ids[label],
        model=MODEL,
    )
    job_ids[label] = fine_tune_response.id
    print(f"🆔 Job ID for {label}: {job_ids[label]}")
    print(f"📍 Status: {fine_tune_response.status}")


# ------------------ STEP 3: Retrieve final job status and model names ------------------
for label in labels:
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

# ------------------ STEP 4: Export model names to Excel ------------------
export_df = pd.DataFrame(
    {
        "category": labels,
        "fine_tuned_model_name": [
            final_model_names["top"],
            final_model_names["bottom"],
        ],
    }
)
export_path = os.path.join(
    EXPORT_DIR, f"{PLATFORM}_{CONCEPT}_APE_finetuned_model_names.xlsx"
)
export_df.to_excel(export_path, index=False)
print(f"\n📁 Exported model names to {export_path}")
