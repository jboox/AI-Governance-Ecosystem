
import streamlit as st
import pandas as pd
import json, requests

st.set_page_config(page_title="Governance Dashboard (MVP)", layout="wide")
st.title("AI Governance Ecosystem – Dashboard (MVP)")

st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ["Skor CKP", "Talent Map"])

# =============== Halaman 1: Skor CKP (NLP) ===============
if page == "Skor CKP":
    st.header("Skor CKP (NLP)")
    st.markdown("Unggah **sample_ckp_labeled.csv** lalu lakukan skoring via API `/nlp/score-ckp`.")
    uploaded = st.file_uploader("Unggah CSV CKP", type=["csv"], key="ckp")
    api_url = st.text_input("URL Backend", value="http://127.0.0.1:8000/nlp/score-ckp")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20))
        payload = [
            {"entry_id": r.entry_id, "uraian_teks": r.uraian_teks, "target": float(r.target), "realisasi": float(r.realisasi)}
            for r in df.head(50).itertuples(index=False)
        ]
        if st.button("Skor via API"):
            try:
                res = requests.post(api_url, json=payload, timeout=15)
                res.raise_for_status()
                scores = pd.DataFrame(res.json())
                st.subheader("Hasil Skor (Top 20)")
                st.dataframe(scores.head(20))
            except Exception as e:
                st.error(f"Error: {e}")

# =============== Halaman 2: Talent Map ===============
if page == "Talent Map":
    st.header("Talent Map")
    st.markdown("Unggah **talent_datamart_sample.csv** lalu panggil API `/ml/talent-score` untuk menghitung **Talent Score** per pegawai. Scatter plot **Mean WQI vs Talent Score** akan ditampilkan.")
    up2 = st.file_uploader("Unggah CSV Talent Datamart", type=["csv"], key="talent")
    api_url2 = st.text_input("URL Backend (talent-score)", value="http://127.0.0.1:8000/ml/talent-score", key="talent_api")
    if up2 is not None:
        tdf = pd.read_csv(up2)
        st.dataframe(tdf.head(10))
        rows = []
        for r in tdf.itertuples(index=False):
            features = {k: float(getattr(r,k)) for k in tdf.columns if k != "pegawai_id"}
            try:
                res = requests.post(api_url2, json={"pegawai_id": getattr(r,"pegawai_id"), "features": features}, timeout=10)
                if res.status_code==200:
                    d = res.json()
                    rows.append({"pegawai_id": d["pegawai_id"], "talent_score": d["talent_score"], **features})
            except Exception as e:
                st.error(f"Gagal memanggil API untuk {getattr(r,'pegawai_id')}: {e}")
        if rows:
            out = pd.DataFrame(rows)
            st.success(f"Skor dihitung untuk {len(out)} pegawai.")
            st.dataframe(out.head(20))
            st.subheader("Scatter: Mean WQI vs Talent Score")
            st.scatter_chart(out, x="mean_wqi", y="talent_score")
        else:
            st.info("Belum ada hasil skor. Pastikan backend berjalan dan file berisi kolom 'pegawai_id', 'mean_wqi', dll.")

st.markdown("---")
st.caption("MVP Dashboard – dua halaman: Skor CKP & Talent Map.")
