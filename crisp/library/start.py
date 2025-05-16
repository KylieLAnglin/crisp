# ------------------ UPDATE HERE ------------------
USER = "Kylie"
PLATFORM = "openai"
CONCEPT = "ncb"
SAMPLE = False
# ------------------ PATHS ------------------


if USER == "Kylie":
    # Kylie
    CODE_DIR = "/Users/kla21002/crisp/crisp/"
    ONEDRIVE = "/Users/kla21002/Library/CloudStorage/OneDrive-UniversityofConnecticut/"

MAIN_DIR = ONEDRIVE + "crisp/"
DATA_DIR = MAIN_DIR + "data/"
RESULTS_DIR = MAIN_DIR + "results/"
RAW_DIR = ONEDRIVE + "Materials for LLM/"

# ------------------ MODEL ------------------
if PLATFORM == "openai":
    MODEL = "gpt-4.1-2025-04-14"

# ------------------ OTHERS ------------------
SEED = 123
