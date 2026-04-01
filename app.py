import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="Dasbor Portofolio DAC", layout="wide")
st.title("Optimasi Portofolio Multiobjektif & Analisis VaR")

# 2. Sidebar untuk Input Parameter Pengguna
st.sidebar.header("Parameter Model")
# Pengguna cukup memilih N, sistem yang akan menyeleksi sahamnya
n_saham = st.sidebar.slider("Pilih Jumlah Saham (N)", min_value=2, max_value=6, value=5)

nilai_k = st.sidebar.number_input("Toleransi Risiko (k)", min_value=0.0, value=10.0, step=1.0)
tingkat_kepercayaan = st.sidebar.number_input("Tingkat Kepercayaan VaR (0-1)", min_value=0.001, max_value=0.999, value=0.95, step=0.01)
horizon_waktu = st.sidebar.number_input("Horizon Waktu (t hari)", min_value=1, value=1, step=1)
modal_awal = st.sidebar.number_input("Modal Awal (V0) - Rp", min_value=0, value=10000000, step=1000000)

# 3. Fungsi Pemrosesan Data (Diperbaiki agar kebal Error)
@st.cache_data
def siapkan_data():
    # Mengatasi masalah delimiter koma atau titik koma
    try:
        df = pd.read_csv("Dataset DAC .csv", delimiter=";")
        if 'Date' not in df.columns:
            df = pd.read_csv("Dataset DAC .csv", delimiter=",")
    except:
        df = pd.read_csv("Dataset DAC .csv", delimiter=",")
    
    # Membersihkan spasi pada nama saham
    df['Stock_Name'] = df['Stock_Name'].str.strip()
    
    # Konversi tanggal yang lebih fleksibel
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Filter hanya tahun 2024
    df_2024 = df[df['Date'].dt.year == 2024]
    
    # Pivot dan hitung Log Return
    df_close = df_2024.pivot(index='Date', columns='Stock_Name', values='Close')
    log_return = np.log(df_close / df_close.shift(1)).dropna()
    return log_return

log_return = siapkan_data()

# 4. TAHAP SELEKSI SAHAM (Algoritma dari Colab)
st.subheader(f"1. Seleksi {n_saham} Saham (Return Positif & Korelasi Rendah)")

# Filter Mean Positif
mean_ret = log_return.mean()
saham_positif = mean_ret[mean_ret > 0].index.tolist()
df_positif = log_return[saham_positif]

# Algoritma Greedy untuk Korelasi (max 0.2)
corr_matrix = df_positif.corr()
# Memulai dari saham dengan return rata-rata tertinggi
terpilih = [df_positif.mean().idxmax()] 

for saham in df_positif.columns:
    if len(terpilih) >= n_saham:
        break
    if saham not in terpilih:
        # Cek korelasi dengan saham yang sudah masuk list
        korelasi_aman = True
        for t in terpilih:
            if corr_matrix.loc[saham, t] > 0.2:
                korelasi_aman = False
                break
        if korelasi_aman:
            terpilih.append(saham)

st.write(f"Saham yang lolos seleksi berdasarkan algoritma: **{', '.join(terpilih)}**")

# Jika sistem gagal menemukan N saham dengan korelasi < 0.2
if len(terpilih) < n_saham:
    st.warning(f"Sistem hanya menemukan {len(terpilih)} saham yang memenuhi kriteria korelasi < 0.2. Analisis dilanjutkan dengan {len(terpilih)} saham tersebut.")

# 5. TAHAP OPTIMASI MULTIOBJEKTIF
ret_terpilih = df_positif[terpilih]
mu = ret_terpilih.mean()
cov_matrix = ret_terpilih.cov()

def optimasi_multiobjektif(mean_returns, cov_mat, k):
    num_assets = len(mean_returns)
    def objective(weights):
        port_return = np.sum(mean_returns * weights)
        port_var = np.dot(weights.T, np.dot(cov_mat, weights))
        return port_var - (k * port_return)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((-1, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

bobot_optimal = optimasi_multiobjektif(mu, cov_matrix, nilai_k)

# Menampilkan Bobot
st.subheader(f"2. Hasil Pembobotan Portofolio (k = {nilai_k})")
df_bobot = pd.DataFrame({
    'Saham': terpilih,
    'Bobot': np.round(bobot_optimal, 4),
    'Persentase': [f"{b*100:.2f}%" for b in bobot_optimal]
})
st.table(df_bobot)

if any(bobot_optimal < 0):
    st.error("⚠️ Peringatan: Terdapat indikasi Short-Selling (bobot bernilai negatif).")

# 6. TAHAP PENGUKURAN RISIKO (VaR Historis)
st.subheader("3. Pengukuran Risiko (Value at Risk - Historical Simulation)")

# Return portofolio harian (weighted sum) sesuai syntax Colab
return_port = ret_terpilih.values @ bobot_optimal

# Hitung percentil
alpha = 1 - tingkat_kepercayaan
percentil = np.percentile(return_port, alpha * 100)

# Rumus VaR sesuai Colab
var_rupiah = modal_awal * abs(percentil) * np.sqrt(horizon_waktu)

col1, col2, col3 = st.columns(3)
col1.metric("Percentil Return", f"{percentil*100:.3f}%")
col2.metric("Horizon Waktu (t)", f"{horizon_waktu} Hari")
col3.metric(f"VaR (Rupiah)", f"Rp {var_rupiah:,.2f}")

st.info(f"Artinya: Ada keyakinan {tingkat_kepercayaan*100:.1f}% bahwa kerugian portofolio tidak akan melebihi Rp {var_rupiah:,.2f} dalam {horizon_waktu} hari ke depan.")
