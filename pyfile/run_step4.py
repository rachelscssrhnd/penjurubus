import os
import json
import joblib
import numpy as np
import pandas as pd
from penjurubus.data_paths import resolve_path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

ROOT = r"D:\penjurubus"
PROC_DIR = os.path.join(ROOT, "data", "processed_v3")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "data", "predictions_v4_search")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = resolve_path(os.path.join(PROC_DIR, "surabaya", "features_v3.parquet"))
VAL_PATH   = resolve_path(os.path.join(PROC_DIR, "yogyakarta", "features_v3.parquet"))
TEGAL_PATH = resolve_path(os.path.join(PROC_DIR, "tegal", "features_v3.parquet"))

TARGET_COL = "is_candidate_stop"

train_df = pd.read_parquet(TRAIN_PATH)
val_df   = pd.read_parquet(VAL_PATH)
tegal_df = pd.read_parquet(TEGAL_PATH)

def split_xy(df):
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    return X, y

X_train, y_train = split_xy(train_df)
X_val, y_val = split_xy(val_df)
X_tegal, y_tegal = split_xy(tegal_df)

drop_cols = [c for c in ["grid_id", "city", "label_method", "geometry", "centroid_x", "centroid_y"] if c in X_train.columns]
X_train = X_train.drop(columns=drop_cols, errors="ignore")
X_val = X_val.drop(columns=drop_cols, errors="ignore")
X_tegal = X_tegal.drop(columns=drop_cols, errors="ignore")

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

print("=" * 60)
print("STEP 4 — Model Search + Threshold Tuning")
print("=" * 60)
print(f"Train shape: {X_train.shape}")
print(f"Val shape  : {X_val.shape}")
print(f"Tegal score: {X_tegal.shape}")
print(f"Positive train rate: {y_train.mean() * 100:.1f}%")
print(f"Numeric cols: {num_cols}")

for c in num_cols:
    med = X_train[c].median()
    X_train[c] = X_train[c].fillna(med)
    X_val[c] = X_val[c].fillna(med)
    X_tegal[c] = X_tegal[c].fillna(med)

for c in cat_cols:
    mode = X_train[c].mode(dropna=True)
    fill = mode.iloc[0] if len(mode) else "missing"
    X_train[c] = X_train[c].fillna(fill).astype(str)
    X_val[c] = X_val[c].fillna(fill).astype(str)
    X_tegal[c] = X_tegal[c].fillna(fill).astype(str)

for c in num_cols:
    mu = X_train[c].mean()
    sd = X_train[c].std(ddof=0)
    sd = 1.0 if pd.isna(sd) or sd == 0 else sd
    X_train[c] = (X_train[c] - mu) / sd
    X_val[c] = (X_val[c] - mu) / sd
    X_tegal[c] = (X_tegal[c] - mu) / sd

for c in cat_cols:
    uniq = list(pd.Series(X_train[c].unique()).dropna())
    if len(uniq) <= 2:
        mapping = {v: i for i, v in enumerate(sorted(uniq))}
        X_train[c] = X_train[c].map(mapping).fillna(0).astype(float)
        X_val[c] = X_val[c].map(mapping).fillna(0).astype(float)
        X_tegal[c] = X_tegal[c].map(mapping).fillna(0).astype(float)
    else:
        X_train = X_train.drop(columns=[c])
        X_val = X_val.drop(columns=[c])
        X_tegal = X_tegal.drop(columns=[c])

final_cols = X_train.columns.tolist()
X_val = X_val.reindex(columns=final_cols, fill_value=0)
X_tegal = X_tegal.reindex(columns=final_cols, fill_value=0)

models = {
    "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    "gboost": GradientBoostingClassifier(random_state=42),
    "extra_trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
}

try:
    models["svm"] = SVC(probability=True, class_weight="balanced", random_state=42)
except Exception:
    pass

def eval_thr(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "cm": confusion_matrix(y_true, pred).tolist()
    }

def best_threshold(y_true, prob):
    grid = np.linspace(0.05, 0.95, 37)
    best_t, best_f1 = 0.5, -1
    for t in grid:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1

results = []
for name, model in models.items():
    print(f"\n[TRAIN] {name}")
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else None
    thr, thr_f1 = best_threshold(y_val, val_prob)
    val_eval = eval_thr(y_val, val_prob, thr)
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Best threshold: {thr:.2f}")
    print(f"  Val F1: {val_eval['f1']:.4f} | P: {val_eval['precision']:.4f} | R: {val_eval['recall']:.4f}")
    results.append({
        "name": name,
        "model": model,
        "val_auc": val_auc,
        "threshold": thr,
        "val_eval": val_eval,
        "val_prob": val_prob
    })

best = max(results, key=lambda x: (x["val_auc"] if x["val_auc"] is not None else -1, x["val_eval"]["f1"]))
best_model = best["model"]
best_name = best["name"]
best_thr = best["threshold"]

print(f"\n[BEST MODEL] {best_name} | val AUC={best['val_auc']:.4f} | thr={best_thr:.2f}")
print("[Validation metrics]")
print(f"  Accuracy : {best['val_eval']['accuracy']:.4f}")
print(f"  Precision: {best['val_eval']['precision']:.4f}")
print(f"  Recall   : {best['val_eval']['recall']:.4f}")
print(f"  F1       : {best['val_eval']['f1']:.4f}")
print(f"  ROC-AUC  : {best['val_eval']['roc_auc']:.4f}")
print(np.array(best["val_eval"]["cm"]))

# score tegal
tegal_prob = best_model.predict_proba(X_tegal)[:, 1]
tegal_pred_thr = (tegal_prob >= best_thr).astype(int)

if tegal_pred_thr.sum() == 0:
    top_n = max(10, int(len(tegal_prob) * 0.05))
    order = np.argsort(-tegal_prob)
    tegal_pred = np.zeros_like(tegal_pred_thr)
    tegal_pred[order[:top_n]] = 1
    rule = f"top_{top_n}"
else:
    tegal_pred = tegal_pred_thr
    rule = f"thr_{best_thr:.2f}"

tegal_scored = tegal_df.copy()
tegal_scored["score"] = tegal_prob
tegal_scored["pred_label"] = tegal_pred
tegal_scored["selected_rule"] = rule
tegal_scored = tegal_scored.sort_values("score", ascending=False).reset_index(drop=True)

tegal_scored.to_pickle(os.path.join(OUT_DIR, "tegal_scored_all.pkl"))
tegal_scored.head(20).to_csv(os.path.join(OUT_DIR, "tegal_top20.csv"), index=False)
tegal_scored.head(50).to_csv(os.path.join(OUT_DIR, "tegal_top50.csv"), index=False)

joblib.dump(best_model, os.path.join(MODEL_DIR, "step4_best_model.joblib"))

summary = {
    "best_model": best_name,
    "best_threshold": best_thr,
    "selected_rule_tegal": rule,
    "validation_metrics": best["val_eval"],
    "train_size": int(len(train_df)),
    "val_size": int(len(val_df)),
    "tegal_size": int(len(tegal_df)),
    "tegal_positive_count": int(tegal_pred.sum()),
    "features": final_cols
}

with open(os.path.join(MODEL_DIR, "step4_best_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n[Tegal scoring]")
print(f"  Rule: {rule}")
print(f"  Positive candidates: {int(tegal_pred.sum())} / {len(tegal_pred)}")
print(f"  Top20 saved: {os.path.join(OUT_DIR, 'tegal_top20.csv')}")
print(f"  Top50 saved: {os.path.join(OUT_DIR, 'tegal_top50.csv')}")
print("\nSTEP 4 selesai")