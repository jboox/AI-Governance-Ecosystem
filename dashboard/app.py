import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ============ App Config ============
st.set_page_config(page_title="Governance Dashboard (MVP)", layout="wide")
st.title("AI Governance Ecosystem – Dashboard (MVP)")

# ============ Helpers ============
def get_health(backend_base: str):
    """Ping /health and return dict or None."""
    url = backend_base.rstrip("/") + "/health"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase & strip spaces; map common aliases -> canonical names."""
    df = df.copy()
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "avg_wqi": "mean_wqi",
        "wqi_mean": "mean_wqi",
        "mean wqi": "mean_wqi",
        "mean_wqi_90d": "mean_wqi_90d",  # disimpan bila ada
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df

def ensure_numeric_features(df: pd.DataFrame, exclude=("pegawai_id",)) -> list:
    """Return numeric feature columns excluding given names."""
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def show_model_status(backend_base: str):
    """Small status header for models."""
    health = get_health(backend_base)
    if not health:
        st.warning("Tidak dapat mengakses /health. Pastikan backend berjalan.")
        return
    cols = st.columns(3)
    cols[0].markdown("**Status Backend**")
    nlp_ok = health.get("nlp_model_loaded", False)
    tl_ok = health.get("talent_model_loaded", False)
    cols[1].write(f"NLP Model: {'✅ Aktif' if nlp_ok else '❌ Belum dimuat'}")
    cols[2].write(f"Talent Model: {'✅ Aktif' if tl_ok else '❌ Belum dimuat'}")

# ============ Sidebar ============
st.sidebar.title("Menu")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Skor CKP", "Talent Map", "Collaboration Graph", "Model Loader", "Talent Model Loader"]
)

# ============ 1) Skor CKP ============
if page == "Skor CKP":
    st.header("Skor CKP (NLP)")
    api_base = st.text_input("Base URL Backend", value="http://127.0.0.1:8000")
    show_model_status(api_base)

    uploaded = st.file_uploader("Unggah CSV CKP (contoh: sample_ckp_labeled.csv)", type=["csv"], key="ckp")
    api_url = api_base.rstrip("/") + "/nlp/score-ckp"

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20))
        # Validasi minimal kolom
        required_cols = {"entry_id", "uraian_teks"}
        if not required_cols.issubset(set(df.columns)):
            st.error(f"CSV harus memuat kolom: {sorted(required_cols)}. Kolom saat ini: {list(df.columns)}")
            st.stop()

        # Pilihan berapa baris di-skor
        n_rows = st.slider("Jumlah baris untuk di-skor", min_value=1, max_value=min(500, len(df)), value=min(50, len(df)))
        # Siapkan payload
        payload = []
        for r in df.head(n_rows).itertuples(index=False):
            item = {
                "entry_id": getattr(r, "entry_id"),
                "uraian_teks": getattr(r, "uraian_teks"),
            }
            # target/realisasi opsional, cast jika ada
            if "target" in df.columns:
                val = getattr(r, "target")
                try:
                    item["target"] = float(val)
                except Exception:
                    item["target"] = None
            if "realisasi" in df.columns:
                val = getattr(r, "realisasi")
                try:
                    item["realisasi"] = float(val)
                except Exception:
                    item["realisasi"] = None
            payload.append(item)

        if st.button("Skor via API"):
            try:
                res = requests.post(api_url, json=payload, timeout=30)
                res.raise_for_status()
                scores = pd.DataFrame(res.json())
                st.subheader("Hasil Skor (Top 20)")
                st.dataframe(scores.head(20))
                # opsi simpan
                st.download_button("Unduh hasil (CSV)", data=scores.to_csv(index=False), file_name="ckp_scores.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Gagal memanggil API: {e}")

# ============ 2) Talent Map ============
if page == "Talent Map":
    st.header("Talent Map – Matplotlib dengan Filter")
    api_base = st.text_input("Base URL Backend", value="http://127.0.0.1:8000", key="talent_api_base")
    show_model_status(api_base)

    up2 = st.file_uploader("Unggah CSV Talent Datamart (contoh: talent_datamart_sample.csv)", type=["csv"], key="talent")
    api_url2 = api_base.rstrip("/") + "/ml/talent-score"

    if up2 is not None:
        tdf_raw = pd.read_csv(up2)
        tdf = normalize_columns(tdf_raw)

        if "pegawai_id" not in tdf.columns:
            st.error("CSV harus punya kolom 'pegawai_id'. Kolom saat ini: " + ", ".join(tdf_raw.columns))
            st.stop()

        st.write("Kolom terdeteksi:", list(tdf.columns))
        st.dataframe(tdf.head(10))

        numeric_cols = ensure_numeric_features(tdf, exclude=("pegawai_id",))
        if not numeric_cols:
            st.error("Tidak ada kolom numerik fitur. Pastikan ada, misalnya: mean_wqi, training_hours_180, late_days_30, dll.")
            st.stop()

        # Pilih fitur X dinamis (default: mean_wqi jika ada)
        default_x = "mean_wqi" if "mean_wqi" in tdf.columns else numeric_cols[0]
        x_feature = st.selectbox("Pilih fitur untuk sumbu X", options=numeric_cols,
                                 index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)

        # Panggil API untuk setiap pegawai
        rows = []
        for r in tdf.itertuples(index=False):
            features = {k: float(getattr(r, k)) for k in numeric_cols}
            try:
                res = requests.post(api_url2, json={"pegawai_id": getattr(r, "pegawai_id"), "features": features}, timeout=10)
                if res.status_code == 200:
                    d = res.json()
                    row = {"pegawai_id": d["pegawai_id"], "talent_score": d["talent_score"], **features}
                    rows.append(row)
            except Exception as e:
                st.error(f"Gagal panggil API untuk {getattr(r, 'pegawai_id')}: {e}")

        if not rows:
            st.info("Belum ada hasil skor. Pastikan backend berjalan dan CSV berisi kolom numerik yang sesuai.")
            st.stop()

        out = pd.DataFrame(rows)
        st.success(f"Skor dihitung untuk {len(out)} pegawai.")
        st.write("Kolom hasil (out):", list(out.columns))
        st.dataframe(out.head(20))

        # Fallback: jika mean_wqi tidak ada pada out, merge dari tdf (kalau ada)
        if "mean_wqi" not in out.columns and "mean_wqi" in tdf.columns:
            out = out.merge(tdf[["pegawai_id", "mean_wqi"]], on="pegawai_id", how="left")

        # Validasi kolom untuk plotting
        if x_feature not in out.columns:
            st.error(f"Kolom '{x_feature}' tidak ada di hasil. Kolom tersedia: {list(out.columns)}")
            st.stop()
        if "talent_score" not in out.columns:
            st.error("Kolom 'talent_score' tidak ada di hasil. Periksa backend /ml/talent-score.")
            st.stop()

        # Slider filter
        # --- Slider filter: robust kalau min==max ---
        # Pastikan tidak ada NaN
        out = out.copy()
        out[x_feature] = pd.to_numeric(out[x_feature], errors="coerce").fillna(0.0)
        out["talent_score"] = pd.to_numeric(out["talent_score"], errors="coerce").fillna(0.0)

        x_min, x_max = float(out[x_feature].min()), float(out[x_feature].max())
        ts_min, ts_max = float(out["talent_score"].min()), float(out["talent_score"].max())

        # Slider untuk X (fitur yang dipilih)
        if x_min < x_max:
            fx = st.slider(
                f"Filter {x_feature}",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
            )
        else:
            st.info(f"Rentang {x_feature} konstan: {x_min}. Slider dinonaktifkan.")
            fx = (x_min, x_max)

        # Slider untuk Talent Score
        if ts_min < ts_max:
            fts = st.slider(
                "Filter Talent Score",
                min_value=ts_min,
                max_value=ts_max,
                value=(ts_min, ts_max),
            )
        else:
            st.info(f"Rentang Talent Score konstan: {ts_min}. Slider dinonaktifkan.")
            fts = (ts_min, ts_max)

        # Terapkan filter (aman walau rentang konstan)
        fdf = out[
            (out[x_feature] >= fx[0]) & (out[x_feature] <= fx[1]) &
            (out["talent_score"] >= fts[0]) & (out["talent_score"] <= fts[1])
        ]

        # --- Plot Matplotlib ---
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.scatter(fdf[x_feature], fdf["talent_score"])
        plt.xlabel(x_feature)
        plt.ylabel("Talent Score")
        plt.title(f"Talent Map (Matplotlib) – {x_feature} vs Talent Score")
        st.pyplot(fig)


        # Plot Matplotlib (tanpa style khusus, sesuai aturan)
        fig = plt.figure()
        plt.scatter(fdf[x_feature], fdf["talent_score"])
        plt.xlabel(x_feature)
        plt.ylabel("Talent Score")
        plt.title(f"Talent Map (Matplotlib) – {x_feature} vs Talent Score")
        st.pyplot(fig)

        # opsi unduh hasil
        st.download_button("Unduh hasil Talent Map (CSV)", data=out.to_csv(index=False),
                           file_name="talent_map_scores.csv", mime="text/csv")

# ============ 3) Collaboration Graph ============
if page == "Collaboration Graph":
    st.header("Collaboration Graph")
    api_base = st.text_input("Base URL Backend", value="http://127.0.0.1:8000", key="graph_api_base")
    show_model_status(api_base)

    upg = st.file_uploader("Unggah CSV Graph Metrics (contoh: graph_metrics_sample.csv)", type=["csv"], key="graph")
    if upg is not None:
        gdf = pd.read_csv(upg)
        gdf = normalize_columns(gdf)
        st.dataframe(gdf.head(20))

        # Top 15 Eigenvector
        if "eigenvector" in gdf.columns and "pegawai_id" in gdf.columns:
            st.subheader("Top 15 by Eigenvector Centrality")
            top = gdf.sort_values("eigenvector", ascending=False).head(15)
            st.bar_chart(top.set_index("pegawai_id")["eigenvector"])

        # Top 15 Betweenness
        if "betweenness" in gdf.columns and "pegawai_id" in gdf.columns:
            st.subheader("Top 15 by Betweenness Centrality")
            topb = gdf.sort_values("betweenness", ascending=False).head(15)
            st.bar_chart(topb.set_index("pegawai_id")["betweenness"])

    st.subheader("Cari Pegawai")
    peg_id = st.text_input("Masukkan pegawai_id untuk ringkasan jaringan", value="P001")
    api_g = api_base.rstrip("/") + "/graph/summary"
    if st.button("Ambil Ringkasan Graph"):
        try:
            res = requests.post(api_g, json={"pegawai_id": peg_id}, timeout=10)
            if res.status_code == 200:
                st.json(res.json())
            else:
                st.error(f"Gagal: {res.status_code} • {res.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# ============ 4) Model Loader (NLP) ============
if page == "Model Loader":
    st.header("NLP Model Loader")
    api_base = st.text_input("Base URL Backend", value="http://127.0.0.1:8000", key="nlp_admin_base")
    show_model_status(api_base)

    joblib_file = st.file_uploader("Pilih file nlp_wqi_baseline.joblib", type=["joblib"], key="joblib")
    api_admin = api_base.rstrip("/") + "/admin/upload-model"
    if joblib_file is not None and st.button("Upload NLP Model"):
        files = {"file": (joblib_file.name, joblib_file.getvalue(), "application/octet-stream")}
        try:
            res = requests.post(api_admin, files=files, timeout=30)
            st.json(res.json())
            if res.status_code == 200:
                st.success("Model NLP berhasil diunggah ke backend.")
            else:
                st.error(f"Gagal upload: {res.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

# ============ 5) Talent Model Loader (XGB) ============
if page == "Talent Model Loader":
    st.header("Talent Model Loader (XGBoost)")
    api_base = st.text_input("Base URL Backend", value="http://127.0.0.1:8000", key="talent_admin_base")
    show_model_status(api_base)

    joblib_file2 = st.file_uploader("Pilih file talent_xgb.joblib", type=["joblib"], key="joblib_talent")
    api_admin2 = api_base.rstrip("/") + "/admin/upload-talent-model"
    if joblib_file2 is not None and st.button("Upload Talent Model"):
        files = {"file": (joblib_file2.name, joblib_file2.getvalue(), "application/octet-stream")}
        try:
            res = requests.post(api_admin2, files=files, timeout=30)
            st.json(res.json())
            if res.status_code == 200:
                st.success("Talent model berhasil diunggah ke backend.")
            else:
                st.error(f"Gagal upload: {res.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("MVP Dashboard – Skor CKP • Talent Map • Collaboration Graph • Model Loader • Talent Model Loader")
