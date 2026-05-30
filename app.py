import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# KONFIGURASI HALAMAN
# ======================

st.set_page_config(
    page_title="Water Potability Dashboard",
    page_icon="💧",
    layout="wide"
)

# ======================
# LOAD DATA
# ======================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
df = pd.read_csv("water_potability.csv")

# ======================
# SIDEBAR
# ======================

st.sidebar.title("💧 Informasi Proyek")

st.sidebar.info("""
Prediksi Kelayakan Air Minum

Model:
Random Forest

Dataset:
Water Potability Dataset
""")

st.sidebar.metric(
    label="Akurasi Model",
    value="60.98%"
)

# ======================
# HEADER
# ======================

st.title("💧 Water Potability Dashboard")

st.markdown("""
Dashboard Machine Learning untuk menganalisis kualitas air dan memprediksi apakah air layak diminum.
""")

colA, colB = st.columns(2)

with colA:
    st.metric("Jumlah Data", len(df))

with colB:
    st.metric("Jumlah Fitur", 9)

st.markdown("---")

# ======================
# TABS
# ======================

tab1, tab2, tab3 = st.tabs([
    "💧 Prediksi",
    "📊 Preliminary Analysis",
    "🔥 Correlation Analysis"
])

# ==================================================
# TAB 1 - PREDIKSI
# ==================================================

with tab1:

    st.subheader("Prediksi Kelayakan Air Minum")

    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input("pH", value=7.0)
        Hardness = st.number_input("Hardness", value=200.0)
        Solids = st.number_input("Solids", value=20000.0)
        Chloramines = st.number_input("Chloramines", value=7.0)
        Sulfate = st.number_input("Sulfate", value=300.0)

    with col2:
        Conductivity = st.number_input("Conductivity", value=400.0)
        Organic_carbon = st.number_input("Organic Carbon", value=10.0)
        Trihalomethanes = st.number_input("Trihalomethanes", value=70.0)
        Turbidity = st.number_input("Turbidity", value=4.0)

    if st.button("🔍 Prediksi"):

        data = np.array([[
            ph,
            Hardness,
            Solids,
            Chloramines,
            Sulfate,
            Conductivity,
            Organic_carbon,
            Trihalomethanes,
            Turbidity
        ]])

        data_scaled = scaler.transform(data)

        hasil = model.predict(data_scaled)

        if hasil[0] == 1:
            st.success("✅ AIR LAYAK MINUM")
            st.balloons()
        else:
            st.error("❌ AIR TIDAK LAYAK MINUM")

# ==================================================
# TAB 2 - PRELIMINARY ANALYSIS
# ==================================================

with tab2:

    st.subheader("Distribusi Air Layak vs Tidak Layak")

    fig1, ax1 = plt.subplots(figsize=(7,4))

    sns.countplot(
        x="Potability",
        data=df,
        palette="Set2",
        ax=ax1
    )

    ax1.set_title(
        "Perbandingan Total Data Air Layak vs Tidak Layak"
    )

    st.pyplot(fig1)

    st.subheader("Distribusi Hardness")

    fig2, ax2 = plt.subplots(figsize=(7,4))

    sns.histplot(
        df["Hardness"],
        kde=True,
        color="purple",
        ax=ax2
    )

    st.pyplot(fig2)

# ==================================================
# TAB 3 - CORRELATION
# ==================================================

with tab3:

    st.subheader("Heatmap Korelasi")

    fig3, ax3 = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        ax=ax3
    )

    st.pyplot(fig3)

    st.subheader("Boxplot Hardness vs Potability")

    fig4, ax4 = plt.subplots(figsize=(7,4))

    sns.boxplot(
        x="Potability",
        y="Hardness",
        data=df,
        ax=ax4
    )

    st.pyplot(fig4)

# ======================
# FOOTER
# ======================

st.markdown("---")

st.caption(
    "Water Potability Prediction Dashboard | Streamlit + Random Forest"
)