import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="Dasbor Portofolio DAC", layout="wide", initial_sidebar_state="expanded")

# Kustomisasi CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50;}
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Optimasi Portofolio Multiobjektif & Analisis VaR")
st.markdown("---")

# 2. Sidebar dengan Form Input dan Penjelasan
st.sidebar.header("⚙️ Parameter Model")

with st.sidebar.form(key='form_analisis'):
    n_saham = st.slider("Jumlah Saham (N)", min_value=2, max_value=6, value=5)
    nilai_k = st.number_input("Toleransi Risiko (k)", min_value=0.01, value=10.0, step=1.0)
    
    st.markdown("---")
    tingkat_kepercayaan = st.number_input("Tingkat Kepercayaan VaR (0-1)", min_value=0.001, max_value=0.999, value=0.95, step=0.01)
    horizon_waktu = st.number_input("Horizon Waktu (t hari)", min_value=1, value=1, step=1)
    modal_awal = st.number_input("Modal Awal (V0) - Rp", min_value=0, value=10000000, step=1000000)
    
    st.markdown("---")
    allow_short = st.checkbox("Izinkan Short-Selling (Bobot Negatif)", value=False)
    
    submit_button = st.form_submit_button(label='Jalankan Analisis 🚀')

with st.sidebar.expander("📖 Panduan Parameter & Interpretasi"):
    st.write("""
    * **Jumlah Saham (N):** Sistem memfilter saham dengan *mean return* positif, lalu menyeleksi $N$ saham yang memiliki rata-rata korelasi terendah (maksimal 0,2). Tujuannya adalah memastikan aset dalam portofolio tidak bergerak searah secara bersamaan, sehingga risiko sistematis dapat ditekan seoptimal mungkin.
    * **Toleransi Risiko (k):** Parameter penalti terhadap variansi dalam fungsi multiobjektif.
        * **k Besar (misal $\ge 50$):** Investor *Risk Averse* (Penghindar Risiko). Portofolio akan terdiversifikasi ke banyak saham untuk meminimalkan variansi matriks.
        * **k Menengah (misal $2-10$):** Investor *Risk Neutral*. Algoritma menyeimbangkan bobot antara mengejar *return* dan menekan risiko.
        * **k Mendekati 0 (misal $0,01$):** Investor *Risk Seeking* (Pencari Risiko). Algoritma memusatkan bobot (hingga 100%) secara agresif pada saham dengan ekspektasi *return* tertinggi dan mengabaikan nilai variansi.
    * **Short-Selling:** Mengubah batasan matematis komputasi. Jika diizinkan, pencarian nilai minimum global mencakup ruang bilangan negatif ($w_i \in \mathbb{R}$). Artinya, investor dapat meminjam saham untuk dijual kosong, guna mendanai alokasi saham lain yang lebih potensial. Jika dimatikan, pencarian dibatasi ketat pada alokasi beli murni ($0 \le w_i \le 1$).
    * **Interpretasi Data 2024:** Output *return* ekspektasi dan VaR yang dihasilkan merepresentasikan estimasi risiko di masa depan dengan asumsi bahwa kondisi dan volatilitas pasar memiliki distribusi yang persis sama dengan memori volatilitas historis sepanjang tahun 2024.
    """)

# 3. Fungsi Pemrosesan Data
@st.cache_data
def siapkan_data():
    try:
        df = pd.read_csv("Dataset DAC .csv", delimiter=";")
        if 'Date' not in df.columns:
            df = pd.read_csv("Dataset DAC .csv", delimiter=",")
    except:
        df = pd.read_csv("Dataset DAC .csv", delimiter=",")
    
    df['Stock_Name'] = df['Stock_Name'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df_2024 = df[df['Date'].dt.year == 2024]
    df_close = df_2024.pivot(index='Date', columns='Stock_Name', values='Close')
    log_return = np.log(df_close / df_close.shift(1)).dropna()
    return log_return

# Eksekusi Utama
if submit_button:
    with st.spinner('Memproses data dan menjalankan optimasi...'):
        log_return = siapkan_data()

        # TAHAP 1: SELEKSI SAHAM
        mean_ret = log_return.mean()
        saham_positif = mean_ret[mean_ret > 0].index.tolist()
        df_positif = log_return[saham_positif]

        corr_matrix = df_positif.corr()
        terpilih = [df_positif.mean().idxmax()] 

        for saham in df_positif.columns:
            if len(terpilih) >= n_saham:
                break
            if saham not in terpilih:
                korelasi_aman = True
                for t in terpilih:
                    if corr_matrix.loc[saham, t] > 0.2:
                        korelasi_aman = False
                        break
                if korelasi_aman:
                    terpilih.append(saham)

        if len(terpilih) < n_saham:
            st.warning(f"Hanya ditemukan {len(terpilih)} saham yang memenuhi kriteria korelasi maksimal 0.2. Analisis dilanjutkan dengan jumlah tersebut.")

        # TAHAP 2: OPTIMASI MULTIOBJEKTIF
        ret_terpilih = df_positif[terpilih]
        mu = ret_terpilih.mean()
        cov_matrix = ret_terpilih.cov()

        def optimasi_multiobjektif(mean_returns, cov_mat, k, allow_short):
            num_assets = len(mean_returns)
            def objective(weights):
                port_return = np.sum(mean_returns * weights)
                port_var = np.dot(weights.T, np.dot(cov_mat, weights))
                # Fungsi Objektif: meminimalkan (k * variansi) - return
                return (k * port_var) - port_return

            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            
            if allow_short:
                bounds = tuple((-2, 2) for _ in range(num_assets))
            else:
                bounds = tuple((0, 1) for _ in range(num_assets))
                
            initial_weights = np.array([1/num_assets] * num_assets)
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x

        bobot_optimal = optimasi_multiobjektif(mu, cov_matrix, nilai_k, allow_short)

        # UI LAYOUT: Hasil Pembobotan
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📌 Hasil Alokasi Bobot")
            df_bobot = pd.DataFrame({
                'Saham': terpilih,
                'Bobot (Desimal)': np.round(bobot_optimal, 4),
                'Persentase': [f"{b*100:.2f}%" for b in bobot_optimal]
            })
            st.dataframe(df_bobot, use_container_width=True)
            st.info(f"**Total Bobot Matematis:** {np.sum(bobot_optimal):.4f} (Mewakili 100% dari modal)")

            if any(bobot_optimal < -0.001):
                st.error("📉 **Status Short-Selling Aktif:** Algoritma merekomendasikan penjualan kosong pada saham berbobot negatif guna memaksimalkan fungsi objektif.")

        with col2:
            st.subheader("📊 Visualisasi Portofolio")
            fig = px.bar(df_bobot, x='Saham', y='Bobot (Desimal)', text='Persentase', 
                         color='Bobot (Desimal)', color_continuous_scale=px.colors.diverging.Tealrose)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # TAHAP 3: PENGUKURAN RISIKO (VaR Historis)
        st.markdown("---")
        st.subheader("🛡️ Pengukuran Risiko (Value at Risk - Historical Simulation)")

        return_port = ret_terpilih.values @ bobot_optimal
        alpha = 1 - tingkat_kepercayaan
        percentil = np.percentile(return_port, alpha * 100)
        var_rupiah = modal_awal * abs(percentil) * np.sqrt(horizon_waktu)

        m1, m2, m3 = st.columns(3)
        m1.metric("Percentil Return Aktual", f"{percentil*100:.3f}%")
        m2.metric("Tingkat Kepercayaan", f"{tingkat_kepercayaan*100:.1f}%")
        m3.metric("Potensi Kerugian (VaR)", f"Rp {var_rupiah:,.2f}", delta="Risiko Maksimal", delta_color="inverse")

        st.success(f"**Interpretasi:** Terdapat probabilitas sebesar **{tingkat_kepercayaan*100:.1f}%** bahwa kerugian aktual portofolio ini tidak akan melebihi estimasi **Rp {var_rupiah:,.2f}** dalam **{horizon_waktu} hari** ke depan, dengan asumsi volatilitas pasar sejalan dengan pergerakan tahun 2024.")

else:
    st.info("👈 Silakan atur parameter di bilah sisi kiri, lalu klik **Jalankan Analisis** untuk melihat hasil komputasi model.")
