# Backend API (MVP v0.2)
Endpoints:
- GET /health
- POST /nlp/score-ckp  (menggunakan model joblib jika tersedia; fallback ke heuristik)
- POST /ml/talent-score
- POST /graph/summary

Jalankan:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```