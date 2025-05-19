# run_all.py
import subprocess
import os
from crisp.library import start

print(
    f"Running all scripts for {start.CONCEPT} on {start.PLATFORM} with {start.MODEL} and sample = {start.SAMPLE}."
)


# ------------------ Prevent Sleep ------------------
caffeinate_proc = subprocess.Popen(["caffeinate"])

# ------------------ Change to Project Directory ------------------

os.chdir(start.CODE_DIR)
print("Current working directory:", os.getcwd())

# ------------------ Define Scripts ------------------
script_paths = [
    # 1. Baseline Prompt
    # "1_baseline_prompt/00_baseline_prep.py",
    # "1_baseline_prompt/01_baseline_train.py",
    # "1_baseline_prompt/02_baseline_dev.py",
    # "1_baseline_prompt/03_fewshot_train.py",
    # "1_baseline_prompt/04_fewshot_dev.py",
    # # 2. Automatic Prompt Engineering
    # "2_APE/01_APE_train.py",
    # "2_APE/02_APE_dev.py",
    "2_APE/03_APE_fewshot_train.py",
    "2_APE/04_APE_fewshot_dev.py",
    # 3. Persona
    # "3_persona/01_persona_train.py",
    # "3_persona/02_persona_dev.py",
    "3_persona/03_persona_fewshot_dev.py",
    # 4. Chain of Thought - Zero
    # "4_zero_shot_cot/01_cot_zero_dev.py",
]

# ------------------ Run Scripts ------------------
try:
    for path in script_paths:
        print(f"\n🚀 Running {path}...\n" + "-" * 50)
        result = subprocess.run(["python", path], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {path} completed successfully.\n")
        else:
            print(f"❌ Error in {path}:\n{result.stderr}\n")
            break  # optional: stop if any script fails

finally:
    caffeinate_proc.terminate()
    print("\n💤 caffeinate stopped — system can now sleep.")
