# run_all_hpc.py
import subprocess
import os
from crisp.library import start

print(
    f"Running all scripts for {start.CONCEPT} on {start.PLATFORM} with {start.MODEL} and sample = {start.SAMPLE}."
)


# ------------------ Define Scripts ------------------
script_paths = [
    # 1. Baseline Prompt
    "crisp/1_baseline_prompt/00_baseline_prep.py",
    "crisp/1_baseline_prompt/01_baseline_train.py",
    "crisp/1_baseline_prompt/02_baseline_dev.py",
    "crisp/1_baseline_prompt/03_fewshot_train.py",
    "crisp/1_baseline_prompt/04_fewshot_dev.py",
    # # # 2. Automatic Prompt Engineering
    # "crisp/2_APE/01_APE_train.py",
    # "crisp/2_APE/02_APE_dev.py",
    # "crisp/2_APE/03_APE_fewshot_train.py",
    # "crisp/2_APE/04_APE_fewshot_dev.py",
    # # 3. Persona
    # "crisp/3_persona/01_persona_train.py",
    # "crisp/3_persona/02_persona_dev.py",
    # "crisp/3_persona/03_persona_fewshot_dev.py",
    # # 4. Chain of Thought - Zero
    # "crisp/4_cot/01_cot_zero_dev.py",
    # "crisp/4_cot/02_cot_fewshot_train.py",
    # "crisp/4_cot/03_cot_fewshot_dev.py",
    # # 5. Explaination
    # "crisp/5_explanation/01_explanation_fewshot_train.py",
    # "crisp/5_explanation/02_explanation_fewshot_dev.py",
]

# ------------------ Run Scripts ------------------
for path in script_paths:
    print(f"\n🚀 Running {path}...\n" + "-" * 50)
    result = subprocess.run(["python3", path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {path} completed successfully.\n")
    else:
        print(f"❌ Error in {path}:\n{result.stderr}\n")
        break  # optional: stop if any script fails
