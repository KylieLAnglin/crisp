import os
import time
import pandas as pd
from openai import OpenAI
from crisp.library import secrets, start

# ------------------ SETUP ------------------
client = OpenAI(api_key=secrets.OPENAI_API_KEY)
MODEL = start.MODEL
PLATFORM = "openai"
EXPORT_DIR = start.DATA_DIR + "finetune/"
EXCEL_PATH = os.path.join(EXPORT_DIR, "fine_tuned_model_names.xlsx")
os.makedirs(EXPORT_DIR, exist_ok=True)


# ------------------ FUNCTION: Upload training file ------------------
def upload_training_file(client, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training file not found: {filepath}")
    print(f"📤 Uploading: {filepath}")
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    print(f"✅ Uploaded. File ID: {response.id}")
    return response.id


# ------------------ FUNCTION: Start fine-tune job and wait ------------------
def start_and_track_finetune(client, file_id, model, suffix, poll_interval=30):
    print(f"🚀 Starting fine-tune with suffix '{suffix}'...")
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=suffix,
    )
    job_id = response.id
    print(f"🆔 Job ID: {job_id} | Status: {response.status}")

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"⏳ Status: {job.status}")
        if job.status in ["succeeded", "failed", "cancelled"]:
            break
        time.sleep(poll_interval)

    if job.status == "succeeded":
        print(f"✅ Success. Model: {job.fine_tuned_model}")
        return job_id, job.fine_tuned_model
    else:
        print(f"❌ Failed. Status: {job.status}")
        if hasattr(job, "error"):
            print(f"Error: {job.error.message}")
        return job_id, None


# ------------------ FUNCTION: Save result to Excel ------------------
def save_model_result(suffix, concept, technique, label, job_id, model_name):
    row = pd.DataFrame(
        [
            {
                "suffix": suffix,
                "concept": concept,
                "technique": technique,
                "label": label,
                "job_id": job_id,
                "model_name": model_name,
            }
        ]
    )
    if os.path.exists(EXCEL_PATH):
        existing = pd.read_excel(EXCEL_PATH)
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row
    updated.to_excel(EXCEL_PATH, index=False)
    print(f"📁 Appended result to {EXCEL_PATH}")


# ------------------ MANUAL EXECUTION BLOCKS (NO LOOP) ------------------

# 1. Gratitude - baseline - top
suffix = "openai_gratitude_baseline_top"
path = os.path.join(EXPORT_DIR, "openai_gratitude_baseline_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "gratitude", "baseline", "top", job_id, model_name)

# 2. Gratitude - baseline - bottom
suffix = "openai_gratitude_baseline_bottom"
path = os.path.join(EXPORT_DIR, "openai_gratitude_baseline_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "gratitude", "baseline", "bottom", job_id, model_name)

# 3. Gratitude - APE - top
suffix = "openai_gratitude_APE_top"
path = os.path.join(EXPORT_DIR, "openai_gratitude_APE_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "gratitude", "APE", "top", job_id, model_name)

# 4. Gratitude - APE - bottom
suffix = "openai_gratitude_APE_bottom"
path = os.path.join(EXPORT_DIR, "openai_gratitude_APE_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "gratitude", "APE", "bottom", job_id, model_name)

# 5. Gratitude - persona - top
suffix = "openai_gratitude_persona_top"
path = os.path.join(EXPORT_DIR, "openai_gratitude_persona_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "gratitude", "persona", "top", job_id, model_name)

# 6. Gratitude - persona - bottom
suffix = "openai_gratitude_persona_bottom"
path = os.path.join(EXPORT_DIR, "openai_gratitude_persona_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "gratitude", "persona", "bottom", job_id, model_name)

# 7. NCB - baseline - top
suffix = "openai_ncb_baseline_top"
path = os.path.join(EXPORT_DIR, "openai_ncb_baseline_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "ncb", "baseline", "top", job_id, model_name)

# 8. NCB - baseline - bottom
suffix = "openai_ncb_baseline_bottom"
path = os.path.join(EXPORT_DIR, "openai_ncb_baseline_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "ncb", "baseline", "bottom", job_id, model_name)

# 9. NCB - APE - top
suffix = "openai_ncb_APE_top"
path = os.path.join(EXPORT_DIR, "openai_ncb_APE_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "ncb", "APE", "top", job_id, model_name)

# 10. NCB - APE - bottom
suffix = "openai_ncb_APE_bottom"
path = os.path.join(EXPORT_DIR, "openai_ncb_APE_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "ncb", "APE", "bottom", job_id, model_name)

# 11. NCB - persona - top
suffix = "openai_ncb_persona_top"
path = os.path.join(EXPORT_DIR, "openai_ncb_persona_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "ncb", "persona", "top", job_id, model_name)

# 12. NCB - persona - bottom
suffix = "openai_ncb_persona_bottom"
path = os.path.join(EXPORT_DIR, "openai_ncb_persona_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "ncb", "persona", "bottom", job_id, model_name)

# 13. MM - baseline - top
suffix = "openai_mm_baseline_top"
path = os.path.join(EXPORT_DIR, "openai_mm_baseline_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "mm", "baseline", "top", job_id, model_name)

# 14. MM - baseline - bottom
suffix = "openai_mm_baseline_bottom"
path = os.path.join(EXPORT_DIR, "openai_mm_baseline_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "mm", "baseline", "bottom", job_id, model_name)

# 15. MM - APE - top
suffix = "openai_mm_APE_top"
path = os.path.join(EXPORT_DIR, "openai_mm_APE_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "mm", "APE", "top", job_id, model_name)

# 16. MM - APE - bottom
suffix = "openai_mm_APE_bottom"
path = os.path.join(EXPORT_DIR, "openai_mm_APE_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "mm", "APE", "bottom", job_id, model_name)

# 17. MM - persona - top
suffix = "openai_mm_persona_top"
path = os.path.join(EXPORT_DIR, "openai_mm_persona_finetune_train_top.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "mm", "persona", "top", job_id, model_name)

# 18. MM - persona - bottom
suffix = "openai_mm_persona_bottom"
path = os.path.join(EXPORT_DIR, "openai_mm_persona_finetune_train_bottom.jsonl")
file_id = upload_training_file(client, path)
job_id, model_name = start_and_track_finetune(client, file_id, MODEL, suffix)
save_model_result(suffix, "mm", "persona", "bottom", job_id, model_name)
