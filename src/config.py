import os

# config.py path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(BASE_DIR)

# project_root
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

# print(PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# print(DATA_DIR)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
# print(RAW_DATA_DIR)
# print(PROCESSED_DATA_DIR)