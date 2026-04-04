import os

ENV = os.environ.get("PROJECT_ENV", "local")

# redirecting towards the alan location for the raw data
if ENV == "alan":
    DATA_RAW = "/scratch/users/andyjalloh/Dataset" 
else:
    DATA_RAW = "data/raw"

# once the data will be normalized, maybe augmented... through preprocess.py
DATA_PROCESSED = "/scratch/users/andyjalloh/data_processed" if ENV == "alan" else "data/processed"

WANDB_PROJECT = "tennis-tracking"