import streamlit as st
import pandas as pd
import numpy as np

# Bagian Judul Halaman
st.set_page_config(page_title="Dasbor Portofolio DAC", layout="wide")
st.title("Optimasi Portofolio MVEP dan Perhitungan VaR")
st.write("Aplikasi analisis saham untuk Data Analysis Competition Matematika Fair 2026.")

# Bagian Sidebar untuk Input Pengguna
st.sidebar.header("Parameter Model")
n_saham = st.sidebar.slider("Jumlah Saham (N)", min_value=2, max_value=6, value=5)
nilai_k = st.sidebar.number_input("Masukkan nilai toleransi risiko (k)", value=10.0)
tingkat_var = st.sidebar.selectbox("Tingkat Kepercayaan VaR", [0.95, 0.99])

# Memuat Data
@st.cache_data
def load_data():
    # Membaca file CSV yang ada di folder yang sama
    df = pd.read_csv("Dataset DAC .csv", delimiter=";")
    return df

data_saham = load_data()

# Menampilkan Data Dasar di Web
st.subheader("1. Tinjauan Data Mentah")
st.dataframe(data_saham.head())

# Tempat untuk logika optimasi multiobjektif Anda nanti
st.subheader("2. Hasil Optimasi dan Pembobotan")
st.write(f"Sistem sedang menghitung bobot untuk {n_saham} saham dengan parameter k = {nilai_k}...")

# Tempat untuk logika peringatan short-selling
# if bobot_negatif:
#     st.warning("Peringatan: Terdapat indikasi Short-Selling pada bobot yang dihasilkan.")