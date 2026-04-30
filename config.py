import os

ROOT = os.environ.get("PENJURUBUS_ROOT", r"D:/penjurubus")
MODEL_DIR = os.path.join(ROOT, "models")
PROC_DIR = os.path.join(ROOT, "data", "processed_v3")
PRED_DIR = os.path.join(ROOT, "data", "predictions_v4_search")

TEGAL_PRED_PATH = os.path.join(PRED_DIR, "tegal", "predictions_v4_search.parquet")

AMLSUBSCRIPTION = os.environ.get("AML_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AML_RESOURCE_GROUP", "rg-penjurubus")
WORKSPACE = os.environ.get("AML_WORKSPACE_NAME", "mlw-penjurubus")

FUNCTIONS_WORKER_RUNTIME = "python"