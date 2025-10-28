# scripts/train_talent_xgb.py (versi auto-build)
import os, json, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dm_path = os.path.join(BASE, "data", "talent_datamart_sample.csv")
ckp_path = os.path.join(BASE, "data", "sample_ckp_labeled.csv")
out_dir = os.path.join(BASE, "backend", "models")
os.makedirs(out_dir, exist_ok=True)

def build_datamart():
    assert os.path.exists(ckp_path), f"CKP tidak ditemukan: {ckp_path}"
    ckp = pd.read_csv(ckp_path)
    feat = ckp.groupby("pegawai_id").agg(
        mean_wqi=("WQI","mean"),
        cnt_entries=("entry_id","count")
    ).reset_index()
    np.random.seed(7)
    feat["late_days_30"] = np.random.poisson(1, size=len(feat))
    feat["training_hours_180"] = np.random.randint(0,40, size=len(feat))
    feat["talent_label_dummy"] = (feat["mean_wqi"] > feat["mean_wqi"].median()).astype(int)
    feat.to_csv(dm_path, index=False)
    return feat

if not os.path.exists(dm_path):
    print("Datamart belum ada. Bangun dari sample_ckp_labeled.csv ...")
    df = build_datamart()
else:
    df = pd.read_csv(dm_path)

label_col = "talent_label_dummy"
if label_col not in df.columns:
    df[label_col] = (df["mean_wqi"] > df["mean_wqi"].median()).astype(int)

feature_cols = [c for c in df.columns if c not in ["pegawai_id", label_col]]
X = df[feature_cols].values
y = df[label_col].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=4, eval_metric='logloss'
)
calibrated = CalibratedClassifierCV(clf, cv=3, method="isotonic")
calibrated.fit(X_tr, y_tr)

proba = calibrated.predict_proba(X_te)[:,1]
auc = roc_auc_score(y_te, proba)
brier = brier_score_loss(y_te, proba)

bundle = {"model": calibrated, "feature_list": feature_cols, "metrics": {"auc": float(auc), "brier": float(brier)}}
joblib.dump(bundle, os.path.join(out_dir, "talent_xgb.joblib"))
print(json.dumps({"auc": float(auc), "brier": float(brier), "features": feature_cols}, indent=2))
