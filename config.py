DATA_DIR = "data"
RESULTS_DIR = "results"

SEED = 42

N_GSM8K = 150
N_MATH = 300                 # split evenly across MATH_LEVELS
MATH_LEVELS = [2, 3, 4, 5]
DEV_FRACTION = 0.30

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300
NUM_CTX = 8192

SOLVER_MODEL = "llama3.2:3b"
SOLVER_TEMP = 0.7 
MAX_TOKENS = 1024
WORKERS = 8

PILOT_K = 5
PILOT_TEMPERATURE = 0.7

PRICES = {
    "llama3.1:8b": (0.05, 0.08),
}