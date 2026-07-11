import os

ROOT = os.environ.get("PENJURUBUS_ROOT", r"D:/penjurubus")
MODEL_DIR = os.path.join(ROOT, "models")
PROC_DIR = os.path.join(ROOT, "data", "processed_v3")
PRED_DIR = os.path.join(ROOT, "data", "predictions_v4_search")

TEGAL_PRED_PATH = os.path.join(PRED_DIR, "tegal", "predictions_v4_search.parquet")