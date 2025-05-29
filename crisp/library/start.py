# ------------------ UPDATE HERE ------------------
# USER = "Kylie", "Brittney", "Claudia", "HPC"
# PLATFORM = "openai", "llama3.2", "llama3.3", "llama4", "gemma3.12"
# CONCEPT = "mm", "ncb", "goodreads"
# SAMPLE = True, False
import os
USER = "Claudia"                  # "HPC"
print(f"DEBUG: Using USER = {USER}")

PLATFORM = "llama3.2"        # "gemma3.12"
CONCEPT = "anger"               # "ncb"
SAMPLE = False

# ------------------ PATHS ------------------
if USER == "Kylie":
    CODE_DIR = "/Users/kla21002/crisp/crisp/"
    ONEDRIVE = "/Users/kla21002/Library/CloudStorage/OneDrive-UniversityofConnecticut/crisp/"
elif USER == "Brittney":
    CODE_DIR = "/Users/brittneyhernandez/Documents/github/crisp/crisp/"
    ONEDRIVE = "/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/project crisp/Anglin, Kylie's files - crisp/"
elif USER == "Claudia":
    CODE_DIR = "C:/Users/Claudia/Documents/GitHub/crisp/crisp/"
    ONEDRIVE = "C:/Users/Claudia/OneDrive - University of Connecticut/Anglin, Kylie's files - crisp/"
elif USER == "HPC":
    import os
    CODE_DIR = os.getcwd() + "/crisp/"  # don't clone crisp repo in a subfolder on hpc
    ONEDRIVE = os.getcwd() + "/crisp/8_hpc/"

MAIN_DIR = ONEDRIVE
DATA_DIR = MAIN_DIR + "data/"
RESULTS_DIR = MAIN_DIR + "results/"

if USER: # without this is using RAW_DIR even if USER is not set to HPC
    RAW_DIR = os.path.abspath(os.path.join(ONEDRIVE, os.pardir)) + "/Materials for LLM/"

print(f"DEBUG: MAIN_DIR = {MAIN_DIR}")
print(f"DEBUG: DATA_DIR = {DATA_DIR}")
print(f"DEBUG: RESULTS_DIR = {RESULTS_DIR}")

# ------------------ MODEL ------------------
if PLATFORM == "openai":
    MODEL = "gpt-4.1-2025-04-14"
elif PLATFORM == "llama3.2":
    MODEL = "llama3.2:latest"
elif PLATFORM == "llama3.3":
    MODEL = "llama3.3:latest"
elif PLATFORM == "llama4":
    MODEL = "llama4:maverick"
elif PLATFORM == "gemma3.12":
    MODEL = "gemma3:12b"

# ------------------ OTHERS ------------------
SEED = 123
