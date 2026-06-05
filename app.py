import os
import io
import warnings
warnings.filterwarnings("ignore")

os.environ["PYSPARK_PYTHON"]        = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

current_year = datetime.now().year

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FraudShield - Sistem Deteksi Fraud",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS: hanya structural, ikuti light/dark otomatis Streamlit ──
st.markdown("""
<style>
    /* ── Layout helpers ── */
    .block-container {
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    }

    /* ── Risk banners ── */
    .risk-box {
        border-radius: 10px; padding: 1.2rem 1.6rem;
        margin-bottom: 0.8rem; border-left: 5px solid;
    }
    .risk-low    { border-color: #22c55e; background: rgba(34,197,94,.08); }
    .risk-medium { border-color: #f59e0b; background: rgba(245,158,11,.08); }
    .risk-high   { border-color: #ef4444; background: rgba(239,68,68,.08); }
    .risk-title  { font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem; }
    .risk-low    .risk-title { color: #16a34a; }
    .risk-medium .risk-title { color: #d97706; }
    .risk-high   .risk-title { color: #dc2626; }
    .risk-desc   { font-size: 0.88rem; opacity: .8; line-height: 1.5; }

    /* ── Status badge ── */
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .badge-approved  { background: #dcfce7; color: #166534; }
    .badge-monitor   { background: #fef9c3; color: #713f12; }
    .badge-review    { background: #ffedd5; color: #9a3412; }
    .badge-blocked   { background: #fee2e2; color: #991b1b; }

    /* ── Info / alert boxes ── */
    .info-note {
        border-left: 4px solid #3b82f6; padding: 0.8rem 1rem;
        border-radius: 6px; font-size: 0.85rem;
        background: rgba(59,130,246,.07); margin: 0.5rem 0;
    }
    .warn-note {
        border-left: 4px solid #f59e0b; padding: 0.8rem 1rem;
        border-radius: 6px; font-size: 0.85rem;
        background: rgba(245,158,11,.07); margin: 0.5rem 0;
    }
    .err-note {
        border-left: 4px solid #ef4444; padding: 0.8rem 1rem;
        border-radius: 6px; font-size: 0.85rem;
        background: rgba(239,68,68,.07); margin: 0.5rem 0;
    }

    /* ── Feature row (SHAP) ── */
    .feat-row { padding: 0.5rem 0; border-bottom: 1px solid rgba(128,128,128,.15); }
    .feat-bar-bg  { height: 7px; background: rgba(128,128,128,.12);
                    border-radius: 4px; overflow: hidden; margin: 4px 0; }
    .feat-bar-pos { height: 100%; background: #ef4444; border-radius: 4px; }
    .feat-bar-neg { height: 100%; background: #22c55e; border-radius: 4px; }

    /* ── History row ── */
    .hist-row {
        display: flex; align-items: center; gap: 10px;
        padding: 0.6rem 0.8rem; border-radius: 8px;
        border: 1px solid rgba(128,128,128,.15); margin-bottom: 0.4rem;
    }

    /* ── Stat summary card ── */
    .sum-card {
        text-align: center; padding: 1rem;
        border-radius: 10px; border: 1px solid rgba(128,128,128,.15);
    }
    .sum-num { font-size: 2rem; font-weight: 700; }
    .sum-lbl { font-size: 0.78rem; opacity: .65; }

    /* ── Divider line ── */
    .sec-divider {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px;
        opacity: .5; border-bottom: 1px solid rgba(128,128,128,.2);
        padding-bottom: 0.4rem; margin: 1rem 0 0.6rem;
    }

    /* ── Validation error inline ── */
    .val-err { color: #dc2626; font-size: 0.8rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "riwayat_prediksi" not in st.session_state:
    st.session_state["riwayat_prediksi"] = []
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []
if "batch_done" not in st.session_state:
    st.session_state["batch_done"] = False


# ============================================================
# CONSTANTS & HELPERS
# ============================================================
CATEGORICAL_COLS = [
    "merchant_category", "merchant_type", "currency", "country",
    "city_size", "card_type", "device", "channel",
]
NUMERICAL_COLS = [
    "amount", "distance_from_home", "high_risk_merchant",
    "transaction_hour", "weekend_transaction",
    "num_transactions", "total_amount_1h",
    "unique_merchants_1h", "unique_countries_1h", "max_single_amount_1h",
]
ALL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

FEATURE_LABELS = {
    "amount":               "Nominal Transaksi",
    "distance_from_home":   "Jarak dari Lokasi Biasa",
    "high_risk_merchant":   "Merchant Berisiko Tinggi",
    "transaction_hour":     "Jam Transaksi",
    "num_transactions":     "Frekuensi Transaksi (1 jam)",
    "total_amount_1h":      "Total Nilai Transaksi (1 jam)",
    "unique_merchants_1h":  "Merchant Berbeda (1 jam)",
    "unique_countries_1h":  "Negara Berbeda (1 jam)",
    "merchant_category":    "Kategori Merchant",
    "merchant_type":        "Tipe Merchant",
    "currency":             "Mata Uang",
    "country":              "Negara",
    "card_type":            "Tipe Kartu",
    "device":               "Perangkat",
    "channel":              "Saluran Transaksi",
    "city_size":            "Ukuran Kota",
    "weekend_transaction":  "Transaksi Akhir Pekan",
    "max_single_amount_1h": "Transaksi Terbesar (1 jam)",
}

FEATURE_BUSINESS_DESC = {
    "distance_from_home": {
        1: "🚨 Transaksi terjadi jauh dari lokasi biasa pemilik kartu - indikasi kuat unauthorized transaction.",
        0: "✅ Transaksi terjadi di sekitar lokasi biasa pemilik kartu - risiko rendah.",
    },
    "amount": {
        "high": "🚨 Nominal transaksi tidak biasa untuk profil pengguna ini - indikasi aktivitas abnormal.",
        "low":  "✅ Nominal transaksi masih dalam rentang normal profil pengguna.",
    },
    "transaction_hour": {
        "risk": "🚨 Transaksi terjadi pada jam rawan penipuan (00:00–05:00) - pola mencurigakan.",
        "safe": "✅ Jam transaksi dalam rentang normal aktivitas pengguna.",
    },

    "unique_countries_1h": {
        "risk": "🚨 Aktivitas dari beberapa negara berbeda dalam satu jam - indikasi card skimming atau account takeover.",
        "safe": "✅ Aktivitas hanya dari satu negara dalam satu jam - normal.",
    },
    "num_transactions": {
        "risk": "🚨 Frekuensi transaksi sangat tinggi dalam 1 jam - indikasi velocity fraud.",
        "safe": "✅ Frekuensi transaksi dalam rentang normal.",
    },
    "channel": {
        "web":    "ℹ️ Saluran web - perlu verifikasi jika dikombinasikan dengan faktor risiko lain.",
        "mobile": "✅ Saluran mobile app - umumnya lebih terautentikasi.",
        "pos":    "✅ Transaksi fisik (EDC) - risiko lebih rendah untuk card-not-present fraud.",
    },
    "merchant_category": {
        "Travel":          "⚠️ Kategori Travel memiliki fraud rate lebih tinggi dibanding kategori lain.",
        "Entertainment":   "⚠️ Kategori Entertainment sering menjadi target fraud.",
        "default":         "ℹ️ Kategori merchant ini memiliki pola fraud yang umum.",
    },
}


def fl(f): return FEATURE_LABELS.get(f, f)


def risk_level(prob: float):
    """Return (level_str, css_class, color_hex) berdasarkan probability fraud."""
    if prob < 0.30:
        return "Low Risk",    "risk-low",    "#16a34a"
    elif prob < 0.60:
        return "Medium Risk", "risk-medium", "#d97706"
    else:
        return "High Risk",   "risk-high",   "#dc2626"


def recommended_action(prob: float, risk_type: str):
    """Return (action_label, badge_class, deskripsi)."""
    if prob < 0.30:
        return (
            "Approve Transaction", "badge-approved",
            "Transaksi dapat diproses. Tidak ditemukan indikasi fraud yang signifikan.",
        )
    elif prob < 0.50:
        return (
            "Monitor Activity", "badge-monitor",
            "Proses transaksi, namun pantau aktivitas akun dalam 1 jam ke depan.",
        )
    elif prob < 0.70:
        return (
            "Require Verification", "badge-review",
            "Tunda transaksi. Minta verifikasi tambahan (OTP/biometrik) dari pemilik kartu.",
        )
    else:
        return (
            "Block Transaction", "badge-blocked",
            "Blokir transaksi. Hubungi pemilik kartu segera untuk konfirmasi.",
        )


def transaction_status(prob: float):
    """Return (status_label, badge_class)."""
    if prob < 0.30:
        return "Approved",     "badge-approved"
    elif prob < 0.50:
        return "Under Review", "badge-monitor"
    elif prob < 0.70:
        return "Pending Verification", "badge-review"
    else:
        return "Blocked",      "badge-blocked"


def generate_investigation_notes(contrib_df, input_dict, prob_fraud):
    """Generate narasi investigasi berbasis SHAP top features."""
    notes = []
    top3 = contrib_df[contrib_df["SHAP"] > 0].head(3)
    for _, row in top3.iterrows():
        feat, sv, iv = row["Feature"], row["SHAP"], row["InputVal"]
        label = fl(feat)
        if feat == "distance_from_home" and int(float(iv)) == 1:
            notes.append(f"**{label}:** Transaksi dilakukan jauh dari domisili pemilik kartu, meningkatkan kemungkinan unauthorized use (SHAP: +{sv:.3f}).")
        elif feat == "amount":
            notes.append(f"**{label}:** Nominal Rp {float(iv):,.0f} tidak biasa untuk profil ini, meningkatkan skor risiko (SHAP: +{sv:.3f}).")
        elif feat == "transaction_hour":
            notes.append(f"**{label}:** Transaksi jam {int(float(iv)):02d}:00 berada di luar pola normal pengguna (SHAP: +{sv:.3f}).")
        elif feat == "unique_countries_1h":
            notes.append(f"**{label}:** {int(float(iv))} negara berbeda dalam 1 jam — indikasi card cloning atau account takeover (SHAP: +{sv:.3f}).")
        elif feat == "num_transactions":
            notes.append(f"**{label}:** {int(float(iv))} transaksi dalam 1 jam — velocity fraud pattern (SHAP: +{sv:.3f}).")
        elif feat == "total_amount_1h":
            notes.append(f"**{label}:** Total Rp {float(iv):,.0f} dalam 1 jam melebihi threshold normal (SHAP: +{sv:.3f}).")
        else:
            notes.append(f"**{label}** (nilai: {iv}) berkontribusi meningkatkan risiko fraud (SHAP: +{sv:.3f}).")
    return notes


def validate_input(d: dict):
    """Return list of (field, message) error tuples."""
    errors = []
    if d.get("amount", 0) <= 0:
        errors.append(("amount", "Nominal transaksi harus lebih dari 0."))
    if not (0 <= d.get("transaction_hour", -1) <= 23):
        errors.append(("transaction_hour", "Jam transaksi harus antara 0–23."))
    if d.get("num_transactions", -1) < 0:
        errors.append(("num_transactions", "Jumlah transaksi tidak boleh negatif."))
    if d.get("total_amount_1h", -1) < 0:
        errors.append(("total_amount_1h", "Total nilai tidak boleh negatif."))
    if d.get("unique_countries_1h", -1) < 0:
        errors.append(("unique_countries_1h", "Jumlah negara tidak boleh negatif."))
    return errors


def feature_business_insight(feat, shap_val, input_val):
    """Return narasi bisnis satu kalimat berdasarkan fitur dan nilai SHAP."""
    try:
        iv_f = float(input_val)
        iv_i = int(iv_f)
    except Exception:
        iv_f, iv_i = None, None

    if shap_val <= 0:
        return None  # hanya tampilkan yang menaikkan risiko

    if feat == "distance_from_home":
        desc = FEATURE_BUSINESS_DESC["distance_from_home"]
        return desc.get(iv_i, desc[1])
    elif feat == "amount":
        return FEATURE_BUSINESS_DESC["amount"]["high"]
    elif feat == "transaction_hour":
        if iv_i is not None and (iv_i <= 5 or iv_i >= 23):
            return FEATURE_BUSINESS_DESC["transaction_hour"]["risk"]
        return None
    elif feat == "unique_countries_1h":
        if iv_i and iv_i > 1:
            return FEATURE_BUSINESS_DESC["unique_countries_1h"]["risk"]
    elif feat == "num_transactions":
        if iv_i and iv_i > 8:
            return FEATURE_BUSINESS_DESC["num_transactions"]["risk"]
    elif feat == "channel":
        return FEATURE_BUSINESS_DESC["channel"].get(str(input_val), None)
    elif feat == "merchant_category":
        cat_desc = FEATURE_BUSINESS_DESC["merchant_category"]
        return cat_desc.get(str(input_val), cat_desc["default"])
    return None


# ============================================================
# LOAD MODEL — hanya load dari folder, tidak training ulang
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    spark = (
        SparkSession.builder
        .appName("FraudShield Inference")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load PipelineModel yang sudah disave dari notebook
    model = PipelineModel.load("model/gbt_pipeline_fulldata")

    # Metrik aktual dari notebook (hardcoded — sudah diverifikasi)
    gbt_metrics = {
        "AUC-ROC":   0.9925,
        "F1-Score":  0.9610,
        "Recall":    0.9602,
        "Precision": 0.9634,
        "Accuracy":  0.9602,
    }

    with open("model/shap_model.pkl", "rb") as f: model_sklearn = pickle.load(f)
    with open("model/scaler.pkl",     "rb") as f: scaler_shap   = pickle.load(f)
    with open("model/encoders.pkl",   "rb") as f: encoders_shap = pickle.load(f)
    explainer = shap.TreeExplainer(model_sklearn)

    return (
        model, spark, explainer,
        scaler_shap, encoders_shap,
        NUMERICAL_COLS, CATEGORICAL_COLS,
        gbt_metrics,
    )


# ── Boot model ──────────────────────────────────────────────
if "model_loaded" not in st.session_state:
    with st.spinner("⚙️ Memuat model deteksi fraud..."):
        try:
            (model, spark, explainer, scaler_shap, encoders_shap,
             numerical_cols, categorical_cols, gbt_metrics) = load_model()
            st.session_state["model_loaded"] = True
            st.session_state["gbt_metrics"]  = gbt_metrics
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            st.info("Pastikan folder `model/gbt_pipeline_fulldata` dan file pkl tersedia.")
            st.stop()
else:
    (model, spark, explainer, scaler_shap, encoders_shap,
     numerical_cols, categorical_cols, gbt_metrics) = load_model()

gbt_m = st.session_state.get("gbt_metrics", {})


# ============================================================
# CORE INFERENCE FUNCTIONS
# ============================================================
def predict_one(input_dict: dict) -> dict:
    """Jalankan prediksi satu transaksi."""
    row = [(
        float(input_dict["amount"]),
        int(input_dict["distance_from_home"]),
        int(input_dict.get("high_risk_merchant", 0)),
        int(input_dict["transaction_hour"]),
        int(input_dict.get("weekend_transaction", 0)),
        int(input_dict["num_transactions"]),
        float(input_dict["total_amount_1h"]),
        int(input_dict["unique_merchants_1h"]),
        int(input_dict["unique_countries_1h"]),
        float(input_dict.get("max_single_amount_1h", input_dict["amount"])),
        str(input_dict["merchant_category"]),
        str(input_dict.get("merchant_type", "online")),
        str(input_dict["currency"]),
        str(input_dict["country"]),
        str(input_dict.get("city_size", "medium")),
        str(input_dict["card_type"]),
        str(input_dict["device"]),
        str(input_dict["channel"]),
    )]
    df_spark = spark.createDataFrame(row, numerical_cols + categorical_cols)
    res = model.transform(df_spark).collect()[0]

    prob_fraud = float(res["probability"][1])
    prob_safe  = float(res["probability"][0])
    risk_score = round(prob_fraud * 100, 1)

    rl, rc, rclr       = risk_level(prob_fraud)
    action, abadge, adesc = recommended_action(prob_fraud, rc)
    status, sbadge     = transaction_status(prob_fraud)

    return {
        "prob_fraud":  prob_fraud,
        "prob_safe":   prob_safe,
        "risk_score":  risk_score,
        "risk_level":  rl,
        "risk_class":  rc,
        "risk_color":  rclr,
        "action":      action,
        "action_badge":abadge,
        "action_desc": adesc,
        "status":      status,
        "status_badge":sbadge,
    }


def compute_shap(input_dict: dict):
    """Hitung SHAP values."""
    X = pd.DataFrame([input_dict])
    for c in categorical_cols:
        le  = encoders_shap[c]
        val = str(X[c].values[0])
        X[c] = le.transform([val]) if val in le.classes_ else [0]
    X[numerical_cols] = scaler_shap.transform(X[numerical_cols])
    shap_vals = explainer.shap_values(X)
    feat_cols = numerical_cols + categorical_cols
    contrib = pd.DataFrame({
        "Feature":  feat_cols,
        "SHAP":     shap_vals[0],
        "InputVal": [input_dict.get(f, "") for f in feat_cols],
    }).sort_values("SHAP", key=abs, ascending=False).reset_index(drop=True)
    return contrib


def render_shap_block(contrib_df, input_dict, pred_result, compact=False):
    """Render SHAP bar chart, feature rows, investigation notes, dan business insights."""
    top_feat = contrib_df.iloc[0]["Feature"]

    # ── Info metodologi ──
    st.markdown(f"""
    <div class="info-note">
        🔬 <b>Metodologi SHAP:</b> Surrogate sklearn GBC (n_estimators=50, max_depth=5),
        dilatih pada 1% sampel training. AUC surrogate = 0.9813.<br>
        🔴 Merah = meningkatkan risiko fraud &nbsp;|&nbsp; 🟢 Hijau = menurunkan risiko fraud<br>
        ⭐ Fitur terpenting: <b>{fl(top_feat)}</b>
    </div>
    """, unsafe_allow_html=True)

    rows = contrib_df.head(8 if compact else 12)
    max_shap = abs(contrib_df["SHAP"].values).max() + 0.01

    # ── Feature contribution rows ──
    st.markdown('<div class="sec-divider">Kontribusi Fitur</div>', unsafe_allow_html=True)
    for _, row in rows.iterrows():
        feat, sv, iv = row["Feature"], row["SHAP"], row["InputVal"]
        pct   = abs(sv) / max_shap * 100
        label = fl(feat)
        sign  = "↑ MENAIKKAN RISIKO" if sv > 0 else "↓ Menurunkan Risiko"
        sclr  = "#dc2626" if sv > 0 else "#16a34a"
        bcls  = "feat-bar-pos"       if sv > 0 else "feat-bar-neg"
        st.markdown(f"""
        <div class="feat-row">
            <div style="display:flex;justify-content:space-between;font-size:0.83rem">
                <b>{label}</b>
                <span style="color:{sclr};font-size:0.72rem">{sign} ({sv:+.3f})</span>
            </div>
            <div class="feat-bar-bg">
                <div class="{bcls}" style="width:{pct:.0f}%"></div>
            </div>
            <div style="font-size:0.72rem;opacity:.65">Nilai: <code>{iv}</code></div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Bar Plot ──
    st.markdown('<div class="sec-divider">SHAP Bar Plot</div>', unsafe_allow_html=True)
    top10   = contrib_df.head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ["#ef4444" if v > 0 else "#22c55e" for v in top10["SHAP"]]
    ax.barh([fl(f) for f in top10["Feature"]], top10["SHAP"],
            color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value", fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(labelsize=8)
    rp = mpatches.Patch(color="#ef4444", label="↑ Meningkatkan risiko")
    gp = mpatches.Patch(color="#22c55e", label="↓ Menurunkan risiko")
    ax.legend(handles=[rp, gp], fontsize=8)
    for bar, val in zip(ax.patches, top10["SHAP"]):
        x  = val + 0.003 if val >= 0 else val - 0.003
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Business Insights ──
    st.markdown('<div class="sec-divider">Interpretasi Bisnis per Fitur</div>',
                unsafe_allow_html=True)
    shown = 0
    for _, row in contrib_df.head(8).iterrows():
        insight = feature_business_insight(row["Feature"], row["SHAP"], row["InputVal"])
        if insight:
            clz = "err-note" if row["SHAP"] > 0 else "info-note"
            st.markdown(f'<div class="{clz}">{insight}</div>', unsafe_allow_html=True)
            shown += 1
        if shown >= 5:
            break

    # ── Why Suspicious? ──
    if pred_result["prob_fraud"] >= 0.30:
        st.markdown('<div class="sec-divider">🔍 Why This Transaction is Suspicious?</div>',
                    unsafe_allow_html=True)
        notes = generate_investigation_notes(contrib_df, input_dict, pred_result["prob_fraud"])
        if notes:
            for n in notes:
                st.markdown(f"- {n}")
        else:
            st.markdown("_Tidak ditemukan faktor dominan yang spesifik. Kombinasi beberapa fitur meningkatkan skor risiko secara keseluruhan._")


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style="margin-bottom:1.2rem">
    <div style="font-size:1.9rem;font-weight:800;letter-spacing:-0.5px">
        🛡️ FraudShield
    </div>
    <div style="opacity:.65;font-size:0.95rem">
        Sistem Deteksi dan Analisis Fraud Transaksi Keuangan Digital
        - GBT + SHAP
    </div>
    <div style="margin-top:0.5rem;font-size:0.75rem;opacity:.5">
        Model Aktif &nbsp;|&nbsp; James Andersen · UMN {current_year}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model metrics bar ──
m1, m2, m3, m4, m5 = st.columns(5)
for col_obj, (lbl, key, color) in zip(
    [m1, m2, m3, m4, m5],
    [("AUC-ROC",   "AUC-ROC",   "#3b82f6"),
     ("F1-Score",  "F1-Score",  "#22c55e"),
     ("Recall",    "Recall",    "#f59e0b"),
     ("Precision", "Precision", "#ec4899"),
     ("Accuracy",  "Accuracy",  "#8b5cf6")],
):
    with col_obj:
        st.metric(lbl, f"{gbt_m.get(key, 0):.4f}")


# ============================================================
# TABS — 5 tab (tab Monitoring dihapus sesuai instruksi)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔍 Prediksi Transaksi",
    "📂 Upload Batch",
    "🧠 Penjelasan AI (SHAP)",
    "🕒 Riwayat Prediksi",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    riwayat = st.session_state["riwayat_prediksi"]
    total   = len(riwayat)
    n_high  = sum(1 for r in riwayat if r.get("risk_level") == "High Risk")
    n_med   = sum(1 for r in riwayat if r.get("risk_level") == "Medium Risk")
    n_low   = sum(1 for r in riwayat if r.get("risk_level") == "Low Risk")

    st.markdown('<div class="sec-divider">📊 Statistik Sesi</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    for col_obj, (lbl, val, color) in zip(
        [d1, d2, d3, d4],
        [("Total Prediksi", total, "#3b82f6"),
         ("High Risk",      n_high, "#ef4444"),
         ("Medium Risk",    n_med,  "#f59e0b"),
         ("Low Risk",       n_low,  "#22c55e")],
    ):
        with col_obj:
            st.markdown(f"""
            <div class="sum-card">
                <div class="sum-num" style="color:{color}">{val}</div>
                <div class="sum-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="sec-divider">Arsitektur Sistem</div>', unsafe_allow_html=True)
        for k, v in [
            ("Dataset",          "Synthetic transactions - 7.483.766 baris (100% sample)"),
            ("Framework",        "Apache Spark (PySpark MLlib) - distributed computing"),
            ("Model Utama",      "GBTClassifier - maxIter=50, maxDepth=7, maxBins=128"),
            ("Imbalance",        "Class Weighting - w_fraud=2.4993, w_nonfraud=0.6250"),
            ("Explainability",   "SHAP TreeExplainer - surrogate GBC (AUC surrogate=0.9813)"),
            ("Deployment",       "Streamlit + PipelineModel.load - inferensi real-time"),
            ("GNN",              "PyTorch Geometric GCN - evaluasi komparatif, tidak di-deploy"),
            ("FL",               "Simulasi skenario terdistribusi - evaluasi saja"),
        ]:
            st.markdown(f"**{k}:** {v}")

    with col_r:
        st.markdown('<div class="sec-divider">Alur Sistem Prediksi</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1", "Input Transaksi",       "Input manual atau upload CSV"),
            ("2", "Preprocessing",         "Feature encoding via PySpark pipeline"),
            ("3", "Prediksi GBT",          "Probabilitas fraud dari model"),
            ("4", "Risk Assessment",       "Low / Medium / High Risk + skor 0–100"),
            ("5", "Recommended Action",    "Approve / Monitor / Verify / Block"),
            ("6", "Transaction Status",    "Approved / Under Review / Blocked"),
            ("7", "SHAP Analysis",         "Kontribusi fitur + narasi investigasi"),
        ]:
            st.markdown(f"**{num}.** **{title}** - {desc}")

    st.markdown("""
    <div class="info-note" style="margin-top:1rem">
        ℹ️ Gunakan tab <b>Prediksi Transaksi</b> untuk transaksi tunggal,
        atau <b>Upload Batch</b> untuk memproses banyak transaksi sekaligus.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI TRANSAKSI TUNGGAL
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-divider">🔍 Input Detail Transaksi Baru</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-note">
        Masukkan detail transaksi baru. Model GBT akan memprediksi tingkat risiko fraud,
        menentukan recommended action, dan memberikan penjelasan AI berbasis SHAP.
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_tunggal", clear_on_submit=False):
        st.markdown("**💳 Informasi Inti Transaksi**")
        c1, c2, c3 = st.columns(3)
        with c1:
            s_amount = st.number_input(
                "Nominal Transaksi (Rp) *",
                min_value=0.0, max_value=100_000_000.0,
                value=500_000.0, step=10_000.0,
                help="⭐ Fitur #2 terpenting (SHAP). Nominal tidak biasa = risiko lebih tinggi."
            )
            s_hour = st.slider(
                "Jam Transaksi (0–23) *", 0, 23, 12,
                help="⭐ Fitur #4 terpenting (SHAP). Jam 00–05 = jam rawan fraud."
            )
        with c2:
            s_channel = st.selectbox(
                "Saluran Transaksi *",
                ["web", "mobile", "pos"],
                format_func=lambda x: {"web": "🌐 Web", "mobile": "📱 Mobile", "pos": "🏪 EDC (POS)"}[x],
                help="⭐ Fitur #3 terpenting (SHAP)."
            )
            s_distance = st.selectbox(
                "Lokasi Transaksi *",
                [0, 1],
                format_func=lambda x: "📌 Dekat lokasi biasa" if x == 0 else "📍 Jauh dari lokasi biasa",
                help="⭐ Fitur #1 terpenting (SHAP). Jauh dari rumah = risiko tinggi."
            )
        with c3:
            s_merchant_cat = st.selectbox(
                "Kategori Merchant",
                ["Retail", "Grocery", "Travel", "Restaurant", "Gas",
                 "Education", "Healthcare", "Entertainment"]
            )

        st.markdown("**📱 Kartu, Perangkat & Lokasi**")
        c4, c5, c6 = st.columns(3)
        with c4:
            s_card = st.selectbox(
                "Tipe Kartu",
                ["Basic Credit", "Gold Credit", "Platinum Credit", "Basic Debit", "Premium Debit"],
                help="⭐ Fitur #7 terpenting (SHAP)."
            )
            s_currency = st.selectbox(
                "Mata Uang", ["USD", "EUR", "AUD", "GBP", "JPY"],
                help="⭐ Fitur #5 terpenting (SHAP)."
            )
        with c5:
            s_device = st.selectbox(
                "Perangkat",
                ["Chrome", "iOS App", "Android", "Android App",
                 "Firefox", "Safari", "Edge", "NFC Payment"],
                help="⭐ Fitur #9 terpenting (SHAP)."
            )
            s_country = st.selectbox(
                "Negara",
                ["USA", "UK", "Australia", "Germany", "Japan", "France", "Canada"],
                help="⭐ Fitur #6 terpenting (SHAP)."
            )
        with c6:
            s_weekend = st.selectbox(
                "Hari Transaksi", [0, 1],
                format_func=lambda x: "Hari Kerja" if x == 0 else "Akhir Pekan"
            )

        st.markdown("**⏱️ Aktivitas Velocity (1 Jam Terakhir)**")
        c7, c8, c9 = st.columns(3)
        with c7:
            s_num_tx   = st.number_input("Jumlah Transaksi *", 0, 50, 3,
                help="⭐ Fitur #8 terpenting. >8 dalam 1 jam = velocity fraud pattern.")
            s_total_1h = st.number_input(
                "Total Nilai (Rp) *", 0.0, 100_000_000.0, 1_500_000.0, step=100_000.0
            )
        with c8:
            s_uniq_merchant = st.number_input("Merchant Berbeda", 0, 50, 2)
            s_uniq_country  = st.number_input(
                "Negara Berbeda *", 0, 10, 1,
                help=">2 negara dalam 1 jam = sangat mencurigakan."
            )
        with c9:
            s_max_single = st.number_input(
                 "Nilai Transaksi Terbesar", 0.0, 100_000_000.0, 500_000.0, step=50_000.0
            )

        submitted = st.form_submit_button(
            "🔍 Analisis Transaksi", use_container_width=True
        )

    # ── Proses setelah submit ──
    if submitted:
        input_dict = {
            "amount":               s_amount,
            "distance_from_home":   s_distance,
            "high_risk_merchant":   0,
            "transaction_hour":     s_hour,
            "weekend_transaction":  s_weekend,
            "num_transactions":     s_num_tx,
            "total_amount_1h":      s_total_1h,
            "unique_merchants_1h":  s_uniq_merchant,
            "unique_countries_1h":  s_uniq_country,
            "max_single_amount_1h": s_max_single,
            "merchant_category":    s_merchant_cat,
            "merchant_type":        "online",
            "currency":             s_currency,
            "country":              s_country,
            "city_size":            "medium",
            "card_type":            s_card,
            "device":               s_device,
            "channel":              s_channel,
        }

        # ── Feature validation ──
        errors = validate_input(input_dict)
        if errors:
            for field, msg in errors:
                st.markdown(f'<div class="err-note">❌ <b>{fl(field)}:</b> {msg}</div>',
                            unsafe_allow_html=True)
        else:
            with st.spinner("🔄 Menganalisis transaksi..."):
                hasil = predict_one(input_dict)

        # Simpan ke riwayat
        st.session_state["riwayat_prediksi"].append({
            "waktu":       datetime.now().strftime("%H:%M:%S"),
            "jenis":       "Tunggal",
            "amount":      s_amount,
            "prob_fraud":  hasil["prob_fraud"],
            "risk_score":  hasil["risk_score"],
            "risk_level":  hasil["risk_level"],
            "risk_class":  hasil["risk_class"],
            "action":      hasil["action"],
            "status":      hasil["status"],
            "input_dict":  input_dict,
        })
        st.session_state["last_single_result"] = hasil
        st.session_state["last_single_input"]  = input_dict

        # ── Risk Banner ──
        desc_map = {
            "Low Risk":    "Transaksi ini tidak menunjukkan pola mencurigakan yang signifikan.",
            "Medium Risk": "Beberapa indikator risiko ditemukan. Direkomendasikan pemantauan lebih lanjut.",
            "High Risk":   "Pola transaksi sangat mirip dengan fraud. Tindakan segera diperlukan.",
        }
        icon_map = {"Low Risk": "✅", "Medium Risk": "⚠️", "High Risk": "🚨"}
        st.markdown(f"""
        <div class="risk-box {hasil['risk_class']}">
            <div class="risk-title">
                {icon_map[hasil['risk_level']]} {hasil['risk_level']}
                &nbsp;–&nbsp; Skor Risiko: {hasil['risk_score']}/100
            </div>
            <div class="risk-desc">{desc_map[hasil['risk_level']]}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric row ──
        r1, r2, r3 = st.columns(3)
        r1.metric("Skor Risiko", f"{hasil['risk_score']}/100")
        r2.metric("Probabilitas Fraud", f"{hasil['prob_fraud']*100:.1f}%")
        r3.metric("Probabilitas Aman",  f"{hasil['prob_safe']*100:.1f}%")

        # ── Recommended Action ──
        st.markdown('<div class="sec-divider">💼 Recommended Action</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <span class="badge {hasil['action_badge']}">{hasil['action']}</span>
        <span style="font-size:0.88rem">{hasil['action_desc']}</span>
        """, unsafe_allow_html=True)

        # ── Transaction Status ──
        st.markdown('<div class="sec-divider">🔖 Transaction Status (Simulasi)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <span class="badge {hasil['status_badge']}">{hasil['status']}</span>
        <span style="font-size:0.85rem;opacity:.7">
            Status ini merupakan simulasi alur fraud detection pada sistem finansial.
        </span>
        """, unsafe_allow_html=True)

        # ── Probability gauge ──
        prob_fraud = hasil["prob_fraud"]
        prob_safe  = hasil["prob_safe"]
        fig_g, ax_g = plt.subplots(figsize=(6, 3))
        theta = np.linspace(np.pi, 0, 300)
        for i in range(len(theta) - 1):
            t   = i / (len(theta) - 2)
            clr = "#22c55e" if t < 0.30 else ("#f59e0b" if t < 0.60 else "#ef4444")
            ax_g.plot(
                [0.8 * np.cos(theta[i]), np.cos(theta[i])],
                [0.8 * np.sin(theta[i]), np.sin(theta[i])],
                color=clr, linewidth=7, alpha=0.9,
            )
        na = np.pi * (1 - hasil["risk_score"] / 100)
        ax_g.annotate("", xy=(0.7 * np.cos(na), 0.7 * np.sin(na)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color="gray", lw=2.5))
        ax_g.plot(0, 0, "o", color="gray", markersize=7, zorder=5)
        ax_g.text(0, 0.12, f"{hasil['risk_score']:.0f}",
                  ha="center", va="center", fontsize=26, fontweight="bold",
                  color=hasil["risk_color"])
        ax_g.text(0, -0.14, "SKOR RISIKO", ha="center", va="center", fontsize=8)
        ax_g.text(-1.0, -0.08, "LOW",    ha="center", color="#22c55e", fontsize=7)
        ax_g.text(0,     1.02, "MEDIUM", ha="center", color="#f59e0b", fontsize=7)
        ax_g.text(1.0,  -0.08, "HIGH",   ha="center", color="#ef4444", fontsize=7)
        ax_g.set_xlim(-1.3, 1.3); ax_g.set_ylim(-0.35, 1.15); ax_g.axis("off")
        plt.tight_layout(pad=0)

        fig_b, ax_b = plt.subplots(figsize=(5, 2.5))
        brs = ax_b.barh(["Aman", "Fraud"], [prob_safe, prob_fraud],
                        color=["#22c55e", "#ef4444"], height=0.45, edgecolor="none")
        ax_b.set_xlim(0, 1)
        ax_b.set_xlabel("Probabilitas", fontsize=8)
        ax_b.tick_params(labelsize=8)
        for bar, val in zip(brs, [prob_safe, prob_fraud]):
            ax_b.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                      f"{val*100:.1f}%", va="center", fontsize=9, fontweight="bold")
        plt.tight_layout()

        g1, g2 = st.columns(2)
        with g1: st.pyplot(fig_g)
        with g2: st.pyplot(fig_b)
        plt.close("all")

        st.markdown("""
        <div class="info-note" style="margin-top:0.8rem">
            💡 Buka tab <b>Penjelasan AI (SHAP)</b> untuk melihat kontribusi fitur,
            investigasi notes, dan interpretasi bisnis dari prediksi ini.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — UPLOAD BATCH
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-divider">📂 Prediksi Batch - Upload CSV</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-note">
        Upload file CSV transaksi baru. Sistem memproses setiap baris,
        menampilkan Risk Level, Recommended Action, dan Transaction Status
        per transaksi, beserta Fraud Alert Summary untuk fraud analyst.
    </div>
    """, unsafe_allow_html=True)

    # ── Template download ──
    contoh = {
        "amount":              [500000, 8500000, 250000, 3200000, 150000],
        "distance_from_home":  [0, 1, 0, 1, 0],
        "transaction_hour":    [14, 2, 10, 23, 9],
        "weekend_transaction": [0, 0, 1, 0, 1],
        "num_transactions":    [3, 15, 2, 7, 1],
        "total_amount_1h":     [1500000, 32000000, 700000, 8000000, 300000],
        "unique_merchants_1h": [2, 10, 1, 4, 1],
        "unique_countries_1h": [1, 3, 1, 2, 1],
        "max_single_amount_1h":[500000, 8500000, 250000, 3200000, 150000],
        "merchant_category":   ["Grocery", "Travel", "Retail", "Entertainment", "Grocery"],
        "merchant_type":       ["online", "online", "offline", "online", "offline"],
        "currency":            ["USD", "JPY", "USD", "EUR", "USD"],
        "country":             ["USA", "Japan", "USA", "Germany", "USA"],
        "city_size":           ["medium", "large", "small", "large", "medium"],
        "card_type":           ["Basic Credit", "Platinum Credit", "Basic Debit",
                                "Gold Credit", "Basic Credit"],
        "device":              ["iOS App", "Chrome", "Android", "Firefox", "iOS App"],
        "channel":             ["mobile", "web", "pos", "web", "mobile"],
    }
    buf = io.StringIO()
    pd.DataFrame(contoh).to_csv(buf, index=False)
    st.download_button("⬇️ Download Template CSV", buf.getvalue(),
                       "template_transaksi.csv", "text/csv")

    uploaded = st.file_uploader("Upload File CSV Transaksi", type=["csv"],
                                help="Format kolom harus sesuai template.")

    if uploaded is not None:
    # Reset state jika file berbeda dari sebelumnya
        if ("last_uploaded_name" not in st.session_state or
                st.session_state.get("last_uploaded_name") != uploaded.name):
            st.session_state["batch_done"]    = False
            st.session_state["batch_results"] = []
            st.session_state["last_uploaded_name"] = uploaded.name

        # Size check
        if uploaded.size > 10 * 1024 * 1024:
            st.markdown('<div class="err-note">❌ File terlalu besar (maks 10 MB).</div>',
                        unsafe_allow_html=True)
            st.stop()

        try:
            df_up = pd.read_csv(uploaded)
            if "high_risk_merchant" not in df_up.columns:
                df_up["high_risk_merchant"] = 0
        except Exception as e:
            st.markdown(f'<div class="err-note">❌ Gagal membaca CSV: {e}</div>',
                        unsafe_allow_html=True)
            st.stop()

        # Column validation
        required = list(contoh.keys())
        missing  = [c for c in required if c not in df_up.columns]
        if missing:
            st.markdown(
                f'<div class="err-note">❌ Kolom tidak ditemukan: {", ".join(missing)}</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        # Row limit
        MAX_ROWS = 1000
        if len(df_up) > MAX_ROWS:
            st.markdown(
                f'<div class="warn-note">⚠️ Maksimal {MAX_ROWS:,} baris. '
                f'File dipotong dari {len(df_up):,} ke {MAX_ROWS:,}.</div>',
                unsafe_allow_html=True,
            )
            df_up = df_up.head(MAX_ROWS)

        # Null check
        if df_up[required].isnull().sum().sum() > 0:
            st.markdown(
                '<div class="warn-note">⚠️ Terdapat nilai kosong — baris tersebut dilewati.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"""
        <div class="info-note">
            ✅ File dimuat - <b>{len(df_up):,} transaksi</b> siap diproses.
        </div>
        """, unsafe_allow_html=True)
        preview_df = df_up.head(5).reset_index(drop=True)
        preview_df.insert(0, "No", range(1, len(preview_df) + 1))

        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Proses Semua Transaksi", use_container_width=True):
            hasil_batch = []
            prog   = st.progress(0)
            status = st.empty()
            n      = len(df_up)

            for idx, baris in df_up.iterrows():
                status.text(f"Memproses {idx+1}/{n}...")
                try:
                    inp = baris.to_dict()
                    for k in ["amount", "total_amount_1h", "max_single_amount_1h"]:
                        if k in inp: inp[k] = float(inp[k])
                    for k in ["distance_from_home", 
                               "transaction_hour", 
                               "weekend_transaction",
                               "num_transactions", 
                               "unique_merchants_1h",
                               "unique_countries_1h"]:
                        if k in inp: inp[k] = int(float(inp[k]))

                    errs = validate_input(inp)
                    if errs:
                        raise ValueError(errs[0][1])

                    h = predict_one(inp)
                    hasil_batch.append({
                        "No":                idx + 1,
                        "Amount":            f"{inp['amount']:,.0f}",
                        "Jam":               inp["transaction_hour"],
                        "Channel":           inp["channel"],
                        "Prob Fraud (%)":    round(h["prob_fraud"] * 100, 1),
                        "Skor Risiko":       h["risk_score"],
                        "Risk Level":        h["risk_level"],
                        "Recommended Action":h["action"],
                        "Transaction Status":h["status"],
                        "_prob_fraud":       h["prob_fraud"],
                        "_risk_class":       h["risk_class"],
                        "_input":            inp,
                    })

                    st.session_state["riwayat_prediksi"].append({
                        "waktu":      datetime.now().strftime("%H:%M:%S"),
                        "jenis":      "Batch",
                        "amount":     inp["amount"],
                        "prob_fraud": h["prob_fraud"],
                        "risk_score": h["risk_score"],
                        "risk_level": h["risk_level"],
                        "risk_class": h["risk_class"],
                        "action":     h["action"],
                        "status":     h["status"],
                        "input_dict": inp,
                    })
                except Exception as ex:
                    hasil_batch.append({
                        "No":                idx + 1,
                        "Amount":            "–",
                        "Jam":               "–",
                        "Channel":           "–",
                        "Prob Fraud (%)":    0,
                        "Skor Risiko":       0,
                        "Risk Level":        "ERROR",
                        "Recommended Action":"–",
                        "Transaction Status":"–",
                        "_prob_fraud":       0,
                        "_risk_class":       "risk-low",
                        "_input":            {},
                    })
                prog.progress((idx + 1) / n)

            status.text("✅ Selesai!")
            st.session_state["batch_results"] = hasil_batch
            st.session_state["batch_done"] = True

            # Refresh dashboard & seluruh session state
            st.rerun()

    # ── Tampilkan hasil batch ──
    if st.session_state.get("batch_done"):
        hasil_batch = st.session_state["batch_results"]
        df_hasil    = pd.DataFrame(hasil_batch)

        n_tot  = len(df_hasil)
        n_high = (df_hasil["Risk Level"] == "High Risk").sum()
        n_med  = (df_hasil["Risk Level"] == "Medium Risk").sum()
        n_low  = (df_hasil["Risk Level"] == "Low Risk").sum()
        avg_p  = df_hasil["_prob_fraud"].mean() * 100
        pct_fr = n_high / n_tot * 100 if n_tot else 0

        # ── Fraud Alert Summary ──
        st.markdown('<div class="sec-divider">📊 Fraud Alert Summary</div>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        for col_obj, (lbl, val, clr) in zip(
            [s1, s2, s3, s4, s5],
            [("Total Transaksi", n_tot, "#3b82f6"),
             ("High Risk",       n_high, "#ef4444"),
             ("Medium Risk",     n_med,  "#f59e0b"),
             ("Low Risk",        n_low,  "#22c55e"),
             ("Avg Prob Fraud",  f"{avg_p:.1f}%", "#8b5cf6")],
        ):
            with col_obj:
                st.markdown(f"""
                <div class="sum-card">
                    <div class="sum-num" style="color:{clr}">{val}</div>
                    <div class="sum-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        # Fraud rate bar
        st.markdown(f"""
        <div style="margin:0.8rem 0">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;opacity:.7">
                <span>High Risk Rate</span><span>{pct_fr:.1f}%</span>
            </div>
            <div style="background:rgba(128,128,128,.15);border-radius:6px;height:10px;overflow:hidden">
                <div style="width:{pct_fr:.1f}%;height:100%;background:#ef4444;border-radius:6px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top Fraud Features Summary ──
        st.markdown('<div class="sec-divider">🔍 Top Fraud Features Summary</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Berdasarkan analisis SHAP pada dataset training, fitur-fitur berikut paling berpengaruh
        terhadap deteksi fraud:
        """)
        top_feats = [
            ("🏠", "Jarak dari Lokasi Biasa",      "Transaksi jauh dari domisili pemilik kartu = risiko tertinggi."),
            ("💰", "Nominal Transaksi",             "Nominal yang tidak biasa untuk profil pengguna meningkatkan skor risiko."),
            ("📱", "Saluran Transaksi",             "Pola saluran yang berbeda dari kebiasaan pengguna = indikasi fraud."),
            ("🕐", "Jam Transaksi",                 "Aktivitas di jam 00:00–05:00 berkorelasi kuat dengan fraud."),
            ("💱", "Mata Uang",                     "Transaksi dalam mata uang tidak biasa menandakan potensi card takeover."),
            ("🌍", "Negara Berbeda (1 jam)",        ">1 negara dalam 1 jam = indikasi card cloning atau account takeover."),
            ("💳", "Tipe Kartu",                    "Jenis kartu tertentu memiliki risiko fraud lebih tinggi."),
            ("🔁", "Frekuensi Transaksi (1 jam)",   "Velocity tinggi dalam waktu singkat = velocity fraud pattern."),
        ]
        for icon, name, desc in top_feats:
            st.markdown(f"- {icon} **{name}:** {desc}")

        # ── Detail tabel ──
        st.markdown('<div class="sec-divider">Detail Hasil per Transaksi</div>',
                    unsafe_allow_html=True)
        cols_show = ["No", "Amount", "Jam", "Channel",
                     "Prob Fraud (%)", "Skor Risiko", "Risk Level",
                     "Recommended Action", "Transaction Status"]
        st.dataframe(
            df_hasil[cols_show].reset_index(drop=True),
            use_container_width=True,
            height=320,
            hide_index=True
        )

        # ── Download dengan fraud label lengkap ──
        df_export = df_hasil[cols_show].copy()
        csv_exp = io.StringIO()
        df_export.to_csv(csv_exp, index=False)
        st.download_button(
            "⬇️ Download Hasil Lengkap (CSV)",
            data=csv_exp.getvalue(),
            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # ── High Risk Alerts ──
        df_high = df_hasil[df_hasil["Risk Level"] == "High Risk"]
        if len(df_high) > 0:
            st.markdown('<div class="sec-divider">🚨 High Risk Transaction Alerts</div>',
                        unsafe_allow_html=True)
            for _, row in df_high.head(10).iterrows():
                st.markdown(f"""
                <div class="err-note">
                    🚨 <b>Transaksi #{row['No']}</b> - 
                    Amount: Rp {row['Amount']} · 
                    Jam: {row['Jam']}:00 · 
                    Prob Fraud: {row['Prob Fraud (%)']}% · 
                    Action: <b>{row['Recommended Action']}</b>
                </div>
                """, unsafe_allow_html=True)

        # ── SHAP pada batch — pilih transaksi ──
        st.markdown('<div class="sec-divider">🧠 SHAP Explainability - Pilih Transaksi</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-note">
            Pilih nomor transaksi dari hasil batch di atas untuk melihat
            penjelasan SHAP, kontribusi fitur, dan narasi investigasi.
        </div>
        """, unsafe_allow_html=True)

        valid_batch = [r for r in hasil_batch if r.get("_input")]
        if valid_batch:
            options = {f"Transaksi #{r['No']} - Risk: {r['Risk Level']} | Prob: {r['Prob Fraud (%)']:.1f}%": i
                       for i, r in enumerate(valid_batch)}
            sel = st.selectbox("Pilih transaksi untuk analisis SHAP:", list(options.keys()))
            if sel and st.button("🔬 Tampilkan SHAP untuk Transaksi Ini"):
                chosen  = valid_batch[options[sel]]
                inp_sel = chosen["_input"]
                with st.spinner("Menghitung SHAP values..."):
                    contrib = compute_shap(inp_sel)
                pred_sel = {
                    "prob_fraud": chosen["_prob_fraud"],
                    "risk_level": chosen["Risk Level"],
                    "risk_color": "#ef4444" if chosen["Risk Level"] == "High Risk"
                                  else "#f59e0b" if chosen["Risk Level"] == "Medium Risk"
                                  else "#22c55e",
                    "risk_score": chosen["Skor Risiko"],
                }
                render_shap_block(contrib, inp_sel, pred_sel)


# ══════════════════════════════════════════════════════════════
# TAB 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-divider">🧠 Penjelasan AI - Kontribusi Fitur (SHAP)</div>',
                unsafe_allow_html=True)

    has_single  = "last_single_result" in st.session_state
    has_riwayat = len(st.session_state["riwayat_prediksi"]) > 0

    if not has_single and not has_riwayat:
        st.markdown("""
        <div class="info-note">
            🧠 Jalankan prediksi terlebih dahulu di tab <b>Prediksi Transaksi</b>
            untuk melihat penjelasan SHAP.
        </div>
        """, unsafe_allow_html=True)
    else:
        if has_single:
            si       = st.session_state["last_single_input"]
            pred_res = st.session_state["last_single_result"]
        else:
            last     = st.session_state["riwayat_prediksi"][-1]
            si       = last["input_dict"]
            pred_res = {
                "prob_fraud": last["prob_fraud"],
                "risk_score": last["risk_score"],
                "risk_level": last.get("risk_level", "–"),
                "risk_color": "#ef4444",
            }

        with st.spinner("🔬 Menghitung SHAP values..."):
            contrib = compute_shap(si)

        # Info prediksi yang dijelaskan
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;
                    padding:0.8rem 1rem;border:1px solid rgba(128,128,128,.2);
                    border-radius:10px;margin:0.8rem 0">
            <div style="font-size:1.5rem;font-weight:700;color:{pred_res.get('risk_color','gray')}">
                {pred_res['risk_score']:.0f}
            </div>
            <div>
                <div style="font-weight:600">Skor Risiko · {pred_res.get('risk_level','–')}</div>
                <div style="font-size:0.78rem;opacity:.65">
                    Probabilitas fraud: {pred_res['prob_fraud']*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_shap_block(contrib, si, pred_res)


# ══════════════════════════════════════════════════════════════
# TAB 5 — RIWAYAT PREDIKSI
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-divider">🕒 Riwayat Prediksi Sesi Ini</div>',
                unsafe_allow_html=True)

    riwayat = st.session_state["riwayat_prediksi"]

    if not riwayat:
        st.markdown("""
        <div class="info-note">
            Belum ada prediksi pada sesi ini.
            Gunakan tab <b>Prediksi Transaksi</b> atau <b>Upload Batch</b> untuk memulai.
        </div>
        """, unsafe_allow_html=True)
    else:
        n_tot  = len(riwayat)
        n_high = sum(1 for r in riwayat if r.get("risk_level") == "High Risk")
        n_med  = sum(1 for r in riwayat if r.get("risk_level") == "Medium Risk")
        n_low  = sum(1 for r in riwayat if r.get("risk_level") == "Low Risk")

        h1, h2, h3, h4 = st.columns(4)
        for col_obj, (lbl, val, clr) in zip(
            [h1, h2, h3, h4],
            [("Total",       n_tot,  "#3b82f6"),
             ("High Risk",   n_high, "#ef4444"),
             ("Medium Risk", n_med,  "#f59e0b"),
             ("Low Risk",    n_low,  "#22c55e")],
        ):
            with col_obj:
                st.markdown(f"""
                <div class="sum-card">
                    <div class="sum-num" style="color:{clr}">{val}</div>
                    <div class="sum-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-divider" style="margin-top:1rem">Detail Riwayat</div>',
                    unsafe_allow_html=True)

        badge_map = {
            "High Risk":   ("badge-blocked",  "HIGH"),
            "Medium Risk": ("badge-review",   "MED"),
            "Low Risk":    ("badge-approved", "LOW"),
        }
        for r in reversed(riwayat[-50:]):
            rl   = r.get("risk_level", "Low Risk")
            bcls, blbl = badge_map.get(rl, ("badge-approved", "OK"))
            clr  = "#ef4444" if rl == "High Risk" else "#f59e0b" if rl == "Medium Risk" else "#22c55e"
            st.markdown(f"""
            <div class="hist-row">
                <span style="font-size:0.72rem;opacity:.5;white-space:nowrap">{r['waktu']}</span>
                <span class="badge {bcls}">{blbl}</span>
                <span style="flex:1;font-size:0.82rem">
                    Rp {r['amount']:,.0f}
                    <span style="font-size:0.72rem;opacity:.5"> · {r['jenis']}</span>
                </span>
                <span style="text-align:right">
                    <span style="color:{clr};font-weight:700">{r['risk_score']:.0f}/100</span>
                    <span style="font-size:0.72rem;opacity:.55"> · {r['prob_fraud']*100:.1f}% fraud</span>
                </span>
                <span style="font-size:0.75rem;opacity:.7">{r.get('action','–')}</span>
            </div>
            """, unsafe_allow_html=True)

        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        df_rw = pd.DataFrame([{
            "Waktu":            r["waktu"],
            "Jenis":            r["jenis"],
            "Amount":           r["amount"],
            "Prob Fraud (%)":   round(r["prob_fraud"] * 100, 1),
            "Skor Risiko":      r["risk_score"],
            "Risk Level":       r.get("risk_level", "–"),
            "Recommended Action": r.get("action", "–"),
            "Transaction Status": r.get("status", "–"),
        } for r in riwayat])

        buf_rw = io.StringIO()
        df_rw.to_csv(buf_rw, index=False)
        st.download_button(
            "⬇️ Export Riwayat CSV",
            data=buf_rw.getvalue(),
            file_name=f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Hapus Riwayat"):
            st.session_state["riwayat_prediksi"] = []
            st.session_state["batch_results"]    = []
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div style="text-align:center;opacity:.45;font-size:0.72rem;
            padding:1.5rem 0 0.5rem;border-top:1px solid rgba(128,128,128,.15);
            margin-top:2rem">
    FraudShield &nbsp;|&nbsp;
    Deteksi Fraud: GBT + SHAP &nbsp;|&nbsp;
    James Andersen · 00000069612 · Sistem Informasi · UMN {current_year}
</div>
""", unsafe_allow_html=True)
