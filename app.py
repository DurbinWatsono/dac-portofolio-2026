import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="GUI Durbin Watsono", layout="wide", initial_sidebar_state="expanded")

# Kustomisasi CSS untuk tampilan formal dan akademik (Times New Roman / Serif)
st.markdown("""
    <style>
    .stApp {
        font-family: 'Times New Roman', Times, serif;
    }
    .main {background-color: #ffffff;}
    h1, h2, h3 {color: #000000;}
    </style>
    """, unsafe_allow_html=True)

st.title("GUI Optimasi Portofolio Multiobjektif dan Analisis VaR")
st.markdown("**Tim Durbin Watsono | Data Analysis Competition - Matematika Fair UNIMED 2026**")
st.markdown("---")

# 2. Sidebar dengan Form Input dan Penjelasan
st.sidebar.header("Parameter Model")

with st.sidebar.form(key='form_analisis'):
    n_saham = st.slider("Jumlah Saham (N)", min_value=2, max_value=6, value=5)
    # Penambahan format="%g" agar tampilan presisi mengikuti input pengguna
    nilai_k = st.number_input("Toleransi Risiko (k)", min_value=1e-9, value=10.0, format="%g")
    
    st.markdown("---")
    tingkat_kepercayaan = st.number_input("Tingkat Kepercayaan VaR", min_value=0.0001, max_value=0.9999, value=0.95, format="%g")
    horizon_waktu = st.number_input("Horizon Waktu (t hari)", min_value=1, value=1, step=1)
    modal_awal = st.number_input("Modal Awal (V0) - Rp", min_value=0.0, value=10000000.0, format="%g")
    
    submit_button = st.form_submit_button(label='Jalankan Analisis')

with st.sidebar.expander("Penjelasan Parameter"):
    st.write("""
    * **Jumlah Saham (N):** GUI melakukan uji normalitas univariat untuk memfilter saham tanpa outlier, lalu memilih saham dengan rata-rata return positif. Selanjutnya, GUI memilih sepasang saham dengan korelasi terkecil sebagai titik awal, lalu menambahkan hingga N saham yang memiliki rata-rata korelasi maksimal 0.2 terhadap saham terpilih. Tujuannya adalah diversifikasi portofolio sehingga risiko seminimum mungkin dam keuntungan semaksimum mungkin.
    * **Toleransi Risiko (k):** Indeks risk aversion (menghindari risiko) yang mengukur toleransi risiko seorang investor.
        * **k Besar (misal 50):** Investor Penghindar Risiko (Risk Averse).
        * **k Menengah (misal 2-10):** Investor Netral Risiko (Risk Neutral).
        * **k Mendekati 0 (misal 0.01):** Investor Berani Risiko (Risk Seeking).
    * **Short-Selling (Trading Limit):** Optimasi multiobjektif ini dapat menghasilkan bobot negatif. Hal ini dinamakan short selling yaitu investor dapat meminjam saham tertentu untuk dijual, lalu dana tersebut dialokasikan ke saham lain. Saham yang dilakukan short selling kelak harus dikembalikan.
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
    
    df_close = df.pivot(index='Date', columns='Stock_Name', values='Close')
    log_return = np.log(df_close / df_close.shift(1)).dropna()
    log_return_2024 = log_return[log_return.index.year == 2024]
    
    return log_return_2024

# Eksekusi Utama
if submit_button:
    with st.spinner('Memproses data dan menjalankan analisis...'):
        log_return = siapkan_data()

        # TAHAP 1: SELEKSI SAHAM
        normal_stocks = []
        for col in log_return.columns:
            stat, p_value = stats.jarque_bera(log_return[col].dropna())
            if p_value > 0.05:
                normal_stocks.append(col)
        
        if len(normal_stocks) >= n_saham:
            df_filtered = log_return[normal_stocks]
        else:
            df_filtered = log_return

        mean_ret = df_filtered.mean()
        saham_positif = mean_ret[mean_ret > 0].index.tolist()
        df_positif = df_filtered[saham_positif]

        corr_matrix = df_positif.corr()
        corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        min_idx = corr_upper.stack().idxmin()
        terpilih = list(min_idx)
        sisa_saham = [s for s in corr_matrix.columns if s not in terpilih]
        
        while sisa_saham and len(terpilih) < n_saham:
            kandidat_scores = {}
            for saham in sisa_saham:
                korelasi = [corr_matrix.loc[saham, s] for s in terpilih]
                avg_corr = np.mean(korelasi)
                kandidat_scores[saham] = avg_corr
                
            saham_baru = min(kandidat_scores, key=kandidat_scores.get)
            avg_terkecil = kandidat_scores[saham_baru]
            
            if avg_terkecil > 0.2:
                break
                
            terpilih.append(saham_baru)
            sisa_saham.remove(saham_baru)

        if len(terpilih) < n_saham:
            st.warning(f"Sistem berhenti pada {len(terpilih)} saham karena kandidat saham selanjutnya memiliki rata-rata korelasi > 0.2. Analisis dilanjutkan dengan {len(terpilih)} saham.")

        st.success(f"Ringkasan Pembentukan: Algoritma telah melakukan seleksi uji normalitas univariat, rata-rata return positif, dan korelasi rata-rata antar saham. Didapat {len(terpilih)} saham ({', '.join(terpilih)}). Optimasi portofolio multiobjektif dilakukan dengan nilai k = {nilai_k}.")

        # TAHAP 2: OPTIMASI MULTIOBJEKTIF
        df_port = df_positif[terpilih]
        
        def hitung_bobot_multiobjektif(df_data, k):
            R = df_data.mean().values
            Sigma = df_data.cov().values
            Sigma_inv = np.linalg.inv(Sigma)
            ones = np.ones(len(R))

            numerator   = (1/(2*k)) * ones @ Sigma_inv @ R - 1
            denominator = (1/(2*k)) * ones @ Sigma_inv @ ones
            lam = numerator / denominator

            w = (1/(2*k)) * Sigma_inv @ (R - lam * ones)
            return w

        bobot_optimal = hitung_bobot_multiobjektif(df_port, nilai_k)

        # UI LAYOUT: Hasil Pembobotan
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Hasil Alokasi Bobot")
            df_bobot = pd.DataFrame({
                'Saham': terpilih,
                'Bobot (Desimal)': np.round(bobot_optimal, 4),
                'Persentase': [f"{b*100:.2f}%" for b in bobot_optimal]
            })
            st.dataframe(df_bobot, use_container_width=True, hide_index=True)
            st.info(f"Total Bobot Matematis: {np.sum(bobot_optimal):.4f} (Mewakili 100% dari modal)")

            if any(bobot_optimal < -0.001):
                st.error("Aktivitas Short-Selling (Trading Limit): Terdapat alokasi bobot negatif. Investor meminjam saham tersebut dari pihak lain untuk dijual dan dananya digunakan untuk mendanai pembelian saham lain yang berbobot positif. Kelak saham tersebut harus dikembalikan beserta imbal hasilnya.")

        with col2:
            st.subheader("Visualisasi Portofolio")
            fig = px.bar(df_bobot, x='Saham', y='Bobot (Desimal)', text='Persentase', 
                         color='Bobot (Desimal)', color_continuous_scale=px.colors.diverging.Tealrose)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # TAHAP 3: PENGUKURAN RISIKO (VaR Historis)
        st.markdown("---")
        st.subheader("Pengukuran Risiko (Value at Risk - Historical Simulation)")

        return_port = df_port.values @ bobot_optimal
        alpha = 1 - tingkat_kepercayaan
        percentil = np.percentile(return_port, alpha * 100)
        var_rupiah = modal_awal * abs(percentil) * np.sqrt(horizon_waktu)

        st.metric("Modal Awal", f"Rp {modal_awal:,.2f}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Persentil Return", f"{percentil*100:.3f}%")
        m2.metric("Tingkat Kepercayaan", f"{tingkat_kepercayaan*100:.1f}%")
        m3.metric("Potensi Kerugian (VaR)", f"Rp {var_rupiah:,.2f}", delta="Risiko Maksimal", delta_color="inverse")

        st.success(f"Interpretasi: Terdapat probabilitas sebesar {tingkat_kepercayaan*100:.1f}% bahwa kerugian aktual portofolio ini tidak akan melebihi estimasi Rp {var_rupiah:,.2f} dalam {horizon_waktu} hari perdagangan ke depan.")

else:
    st.info("Silakan atur parameter di bilah sisi kiri, lalu klik Jalankan Analisis untuk melihat hasil analisis.")
