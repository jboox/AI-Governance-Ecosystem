
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import re, os

app = FastAPI(title="AI Governance – MVP API", version="0.2.0")

class CKPItem(BaseModel):
    entry_id: str
    uraian_teks: str
    target: Optional[float] = None
    realisasi: Optional[float] = None

class ScoreResponse(BaseModel):
    entry_id: str
    work_quality: int
    relevance: int
    impact: int
    evidence: int
    clarity: int
    compliance: int
    wqi: int

def clamp(v, lo=1, hi=5):
    return max(lo, min(hi, v))

def heuristic_score(text: str, target=None, realisasi=None):
    t = text.lower()
    action_words = ["menyusun","menganalisis","mengolah","menyelesaikan","mengembangkan","memverifikasi","menyediakan","merekap"]
    numbers = re.findall(r"\b\d+[\.,]?\d*%?|\b(berkas|indikator|tautan|link)\b", t)
    has_unit = any(w in t for w in ["%", "indikator", "berkas", "tautan", "link", "baseline", "target"])
    length = len(t.split())
    rel = 2 + int(any(a in t for a in action_words)) + int("kinerja" in t or "laporan" in t or "sop" in t)
    impact_kw = ["efisiensi","meningkat","turun","akurat","capaian","realisasi"]
    dmp = 2 + int(len(numbers) >= 1) + int(any(k in t for k in impact_kw))
    bkt = 1 + min(4, len(numbers)//1)
    jls = 2 + int(10 <= length <= 40) + int("," in t or "." in t)
    kpt = 2 + int(target is not None or realisasi is not None) + int(has_unit)
    rel = clamp(rel); dmp = clamp(dmp); bkt = clamp(bkt); jls = clamp(jls); kpt = clamp(kpt)
    avg = (rel + dmp + bkt + jls + kpt)/5.0
    wqi = round((avg - 1)/4 * 100)
    return rel, dmp, bkt, jls, kpt, wqi

# --- Joblib model hook (optional) ---
_model_bundle = None
try:
    import joblib
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "nlp_wqi_baseline.joblib"))
    if os.path.exists(model_path):
        _model_bundle = joblib.load(model_path)  # {'vectorizer','model','bucket_to_score'}
except Exception:
    _model_bundle = None

def score_with_model(text: str, target=None, realisasi=None):
    if _model_bundle is None:
        return heuristic_score(text, target, realisasi)
    vec = _model_bundle['vectorizer']
    clf = _model_bundle['model']
    bucket_to_score = _model_bundle.get('bucket_to_score',{0:10,1:30,2:50,3:70,4:90})
    X = vec.transform([text])
    pred_bucket = int(clf.predict(X)[0])
    wqi = int(bucket_to_score.get(pred_bucket, 50))
    # simple proxies for sub-scores
    t = text.lower()
    rel = 2 + int(any(k in t for k in ["sasaran","target","iku","rencana"])) + int(any(k in t for k in ["laporan","kinerja","sop"]))
    dmp = 2 + int(any(k in t for k in ["efisiensi","meningkat","turun","akurat"])) + int("%" in t)
    bkt = 1 + min(4, len(re.findall(r"\b\d+[\.,]?\d*%?", t)))
    jls = 2 + int(10 <= len(t.split()) <= 40) + int("," in t or "." in t)
    kpt = 2 + int(any(k in t for k in ["realisasi","target","lampiran","berkas","tautan","link"]))
    rel = clamp(rel); dmp = clamp(dmp); bkt = clamp(bkt); jls = clamp(jls); kpt = clamp(kpt)
    return rel, dmp, bkt, jls, kpt, wqi

@app.get("/health")
def health():
    return {"status":"ok","time": datetime.utcnow().isoformat(), "model_loaded": _model_bundle is not None}

@app.post("/nlp/score-ckp", response_model=List[ScoreResponse])
def score_ckp(items: List[CKPItem]):
    results = []
    for it in items:
        rel,dmp,bkt,jls,kpt,wqi = score_with_model(it.uraian_teks, it.target, it.realisasi)
        results.append(ScoreResponse(
            entry_id=it.entry_id,
            work_quality=wqi,
            relevance=rel,
            impact=dmp,
            evidence=bkt,
            clarity=jls,
            compliance=kpt,
            wqi=wqi
        ))
    return results

# ---- Talent score stub ----
class TalentRequest(BaseModel):
    pegawai_id: str
    features: Dict[str, float] = {}

class TalentResponse(BaseModel):
    pegawai_id: str
    talent_score: int
    band: str
    top_factors: Dict[str, float] = {}

@app.post("/ml/talent-score", response_model=TalentResponse)
def talent_score(req: TalentRequest):
    m = req.features.get('mean_wqi', 50.0)
    train = req.features.get('training_hours_180', 0.0)
    late = req.features.get('late_days_30', 0.0)
    score = int(max(0, min(100, 0.6*m + 0.8*train - 2*late)))
    band = 'High' if score>=70 else ('Medium' if score>=40 else 'Emerging')
    return TalentResponse(pegawai_id=req.pegawai_id, talent_score=score, band=band,
                          top_factors={'mean_wqi': m, 'training_hours_180': train, 'late_days_30': late})

# ---- Graph summary stub ----
class GraphQuery(BaseModel):
    pegawai_id: str

class GraphSummary(BaseModel):
    pegawai_id: str
    degree: float
    betweenness: float
    eigenvector: float
    community: int

@app.post("/graph/summary", response_model=GraphSummary)
def graph_summary(q: GraphQuery):
    import pandas as pd
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","data","graph_metrics_sample.csv"))
    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df[df['pegawai_id']==q.pegawai_id]
        if len(row):
            r = row.iloc[0]
            return GraphSummary(pegawai_id=q.pegawai_id, degree=float(r['degree']), betweenness=float(r['betweenness']), eigenvector=float(r['eigenvector']), community=int(r['community']))
    return GraphSummary(pegawai_id=q.pegawai_id, degree=0.0, betweenness=0.0, eigenvector=0.0, community=-1)
