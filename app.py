import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="Dasbor Portofolio DAC", layout="wide")
st.title("Optimasi Portofolio Multiobjektif & Analisis VaR")
st.write("Aplikasi analisis saham interaktif untuk Data Analysis Competition 2026.")

# 2. Sidebar untuk Input Parameter Pengguna
st.sidebar.header("Parameter Model")
# Memilih saham yang akan dianalisis (Contoh 6 saham yang sudah lolos uji di jurnal/notebook)
daftar_saham = ['BBCA ', 'BMRI ', 'BBNI ', 'JSMR ', 'LPKR ', 'INDF ']
saham_terpilih = st.sidebar.multiselect(
    "Pilih Saham untuk Portofolio (Min. 2)", 
    options=daftar_saham, 
    default=daftar_saham[:5]
)

nilai_k = st.sidebar.number_input("Toleransi Risiko (k)", min_value=0.0, value=10.0, step=1.0)
tingkat_kepercayaan = st.sidebar.number_input("Tingkat Kepercayaan VaR (0-1)", min_value=0.001, max_value=0.999, value=0.95, step=0.01)
modal_awal = st.sidebar.number_input("Modal Awal (Rp)", min_value=0, value=10000000, step=1000000)

# 3. Fungsi Pemrosesan Data
@st.cache_data
def siapkan_data():
    df = pd.read_csv("Dataset DAC .csv", delimiter=";")
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
    # Filter hanya tahun 2024
    df_2024 = df[df['Date'].dt.year == 2024]
    
    # Pivot dan hitung Log Return
    df_close = df_2024.pivot(index='Date', columns='Stock_Name', values='Close')
    log_return = np.log(df_close / df_close.shift(1)).dropna()
    return log_return

# 4. Fungsi Optimasi Multiobjektif
def optimasi_multiobjektif(mean_returns, cov_matrix, k):
    num_assets = len(mean_returns)
    
    # Fungsi Objektif: f(w) = -k * E(Rp) + Var(Rp)
    def objective(weights):
        port_return = np.sum(mean_returns * weights)
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        return port_var - (k * port_return)

    # Constraint: Total bobot = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Bounds: Dibuat (-1, 1) agar algoritma bisa menghasilkan bobot negatif (short-selling) jika k terlalu ekstrem
    bounds = tuple((-1, 1) for _ in range(num_assets))
    
    initial_weights = np.array([1/num_assets] * num_assets)
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 5. Eksekusi Utama
if len(saham_terpilih) >= 2:
    log_return = siapkan_data()
    # Filter return hanya untuk saham yang dipilih pengguna
    ret_terpilih = log_return[saham_terpilih]
    
    st.subheader("1. Tinjauan Data Log Return (Tahun 2024)")
    st.dataframe(ret_terpilih.head())
    
    # Menghitung parameter portofolio
    mu = ret_terpilih.mean()
    cov_matrix = ret_terpilih.cov()
    
    # Optimasi
    bobot_optimal = optimasi_multiobjektif(mu, cov_matrix, nilai_k)
    
    # Menampilkan Bobot
    st.subheader("2. Hasil Pembobotan Portofolio")
    df_bobot = pd.DataFrame({
        'Saham': saham_terpilih,
        'Bobot': np.round(bobot_optimal, 4),
        'Persentase': [f"{b*100:.2f}%" for b in bobot_optimal]
    })
    st.table(df_bobot)
    
    # Peringatan Short-Selling
    if any(bobot_optimal < 0):
        st.error("⚠️ Peringatan: Terdapat indikasi Short-Selling (bobot bernilai negatif). Hal ini terjadi karena parameter k terlalu tinggi atau karakteristik saham tidak proporsional.")
    
    # Menghitung VaR dengan Historical Simulation
    port_returns = ret_terpilih.dot(bobot_optimal)
    # Persentil dihitung berdasarkan (1 - tingkat_kepercayaan)
    alpha = 1 - tingkat_kepercayaan
    var_percentile = np.percentile(port_returns, alpha * 100)
    var_rupiah = modal_awal * abs(var_percentile)
    
    st.subheader("3. Pengukuran Risiko (Value at Risk - Historical Simulation)")
    st.write(f"Berdasarkan tingkat kepercayaan **{tingkat_kepercayaan*100:.1f}%**, potensi kerugian maksimum harian adalah:")
    st.metric(label="VaR (Persentase)", value=f"{var_percentile*100:.3f}%")
    st.metric(label="VaR (Rupiah)", value=f"Rp {var_rupiah:,.2f}")

else:
    st.warning("Silakan pilih minimal 2 saham pada sidebar di sebelah kiri untuk memulai analisis.")
