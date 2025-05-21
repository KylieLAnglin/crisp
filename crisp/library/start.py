# ------------------ UPDATE HERE ------------------
# USER = "Kylie", "Brittney", "Claudia", "HPC"
# PLATFORM = "openai", "llama"
# CONCEPT = "mm", "ncb", "goodreads"
# SAMPLE = True, False
USER = "Brittney"
PLATFORM = "llama"
CONCEPT = "mm"
SAMPLE = False

# ------------------ PATHS ------------------
if USER == "Kylie":
    CODE_DIR = "/Users/kla21002/crisp/crisp/"
    ONEDRIVE = "/Users/kla21002/Library/CloudStorage/OneDrive-UniversityofConnecticut/"
if USER == "Brittney":
    CODE_DIR = "/Users/brittneyhernandez/Documents/github/crisp/crisp/"
    ONEDRIVE = "/Users/brittneyhernandez/Library/CloudStorage/OneDrive-UniversityofConnecticut/"
if USER == "Claudia":
    CODE_DIR = "add/path/to/crisp/crisp/repo/"
    ONEDRIVE = "add/path/to/OneDrive/"
if USER == "HPC":
    import os
    CODE_DIR = os.getcwd() + "/crisp/crisp" # don't clone crisp repo in a subfolder on hpc
    ONEDRIVE = os.getcwd() + "/crisp/crisp/hpc/"

MAIN_DIR = ONEDRIVE + "crisp/"
DATA_DIR = MAIN_DIR + "data/"
RESULTS_DIR = MAIN_DIR + "results/"
RAW_DIR = ONEDRIVE + "Materials for LLM/"


# ------------------ MODEL ------------------
if PLATFORM == "openai":
    MODEL = "gpt-4.1-2025-04-14"
if PLATFORM == "llama":
    MODEL = "llama3.2"

# ------------------ OTHERS ------------------
SEED = 123
