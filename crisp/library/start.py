ONEDRIVE = "/Users/kla21002/Library/CloudStorage/OneDrive-UniversityofConnecticut/"


MAIN_DIR = ONEDRIVE + "crisp/"

DATA_DIR = MAIN_DIR + "data/"
RESULTS_DIR = MAIN_DIR + "results/"

RAW_DIR = ONEDRIVE + "Materials for LLM/"

PLATFORM = "openai"

models = {"openai": "gpt-4.1-2025-04-14", "llama": "llama3.2"}

MODEL = models[PLATFORM]

SEED = 123
