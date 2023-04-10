from os import path

SUFFIX_LENGTH = 3
BASE_DIR = path.abspath(__file__)
for i in range(SUFFIX_LENGTH):
    BASE_DIR = path.dirname(BASE_DIR)

DATA_DIR = path.join(BASE_DIR, "data")
RESULT_DIR = path.join(BASE_DIR, "results")

# Used in mlflow to store the code
ARTIFACT_PATH = path.join(BASE_DIR, "code.zip")
IGNORE_FILE_PATTERNS = ["code.zip", "mlrun", "old_", "_old", "OLD_", "_OLD"]
IGNORE_DIRECTORY_PATTERNS = [
    "__pycache__",
    ".git",
    ".venv",
    ".vscode",
    ".trash",
    "mlruns",
    "old",
    "pacoh_old",
    "results",
    "logs",
    "tensorboard",
]

# The device which pytorch uses
device = "cpu"
