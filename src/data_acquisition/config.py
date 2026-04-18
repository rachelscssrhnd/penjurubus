import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
SPLIT_DIR      = os.path.join(BASE_DIR, "data", "split")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")

# Model paths
MODEL_DIR      = os.path.join(BASE_DIR, "models")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

# Split ratio
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
RANDOM_SEED    = 7        

# Studi kasus kota
VALIDATION_CITIES = ["kota surabaya", "kota yogyakarta"]
TEST_CITIES       = ["kota tegal"]

# Grid parameter
GRID_SIZE_M    = 500         # ukuran grid dalam meter (sesuai standar ITDP)
WALK_RADIUS_M  = 500         # radius jalan kaki ke halte

# IPSO-GA parameter
N_PARTICLES    = 50
MAX_ITER       = 1000
C1             = 2.0         # cognitive coefficient
C2             = 2.0         # social coefficient
W_INIT         = 1.0         # initial inertia weight
CROSSOVER_RATE = 0.8
MUTATION_RATE  = 0.1

# Explainability
TOP_N_FEATURES = 3          