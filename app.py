import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- 1. LOAD ASSETS DENGAN CACHING ---
@st.cache_resource
def load_model_and_assets():
    # Menggunakan cache agar file hanya dibaca sekali saat aplikasi start 
    model = joblib.load('final_xgb_model.joblib')
    assets = joblib.load('preprocessing_assets.joblib')
    return model, assets

model, assets = load_model_and_assets()

# --- 2. FUNGSI TRANSFORMASI DATA ---
def transform_user_input(data, assets):
    # Membuat dataframe kosong dengan 70 kolom sesuai blueprint training [cite: 1216]
    input_df = pd.DataFrame(0, index=[0], columns=assets['feature_columns'])
    
    # Mapping fitur numerik utama dengan batas Winsorizing [cite: 923-925, 943]
    input_df['Rainfall'] = np.clip(data['rainfall'], 0, 37.40)
    input_df['Humidity3pm'] = data['humidity_3pm'] # Top Predictor [cite: 1316]
    input_df['WindGustSpeed'] = np.clip(data['wind_gust_speed'], 15.0, 81.0)
    input_df['Sunshine'] = data['sunshine']
    input_df['Pressure9am'] = data['pressure_9am']
    input_df['MinTemp'] = data['min_temp']
    input_df['MaxTemp'] = data['max_temp']
    
    # Feature Engineering Waktu [cite: 535-540]
    input_df['Year'] = data['date'].year
    input_df['Month'] = data['date'].month
    input_df['Day'] = data['date'].day
    
    # Encoding Kategorikal [cite: 556, 614]
    input_df['RainToday'] = assets['rain_mapping'][data['rain_today']]
    input_df['WindGustDir_Encoded'] = assets['wind_mapping'][data['wind_gust_dir']]
    
    # Imputasi untuk fitur yang tidak ada di UI [cite: 994]
    input_df['WindDir9am_Encoded'] = assets['imputation_values'].get('WindDir9am_Encoded', 0)
    input_df['WindDir3pm_Encoded'] = assets['imputation_values'].get('WindDir3pm_Encoded', 6)

    # One-Hot Encoding Lokasi [cite: 1209-1215]
    loc_col = f"Location_{data['location']}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1
        
    return input_df

# --- 3. USER INTERFACE (UI) ---
st.title("Aplikasi Prediksi Hujan Australia")
st.write("Gunakan menu di samping untuk memasukkan data cuaca.")

with st.sidebar:
    st.header("Input Parameter")
    date_in = st.date_input("Tanggal Observasi", datetime.now())
    loc_in = st.selectbox("Lokasi Stasiun", assets['locations'])
    hum_in = st.slider("Kelembaban 3 PM (%)", 0.0, 100.0, 50.0)
    rain_in = st.number_input("Curah Hujan Hari Ini (mm)", 0.0, 37.4, 0.0)
    sun_in = st.slider("Jam Sinar Matahari", 0.0, 15.0, 7.0)
    wind_s_in = st.slider("Kecepatan Angin (km/jam)", 15.0, 81.0, 40.0)
    wind_d_in = st.selectbox("Arah Angin", list(assets['wind_mapping'].keys()))
    press_in = st.number_input("Tekanan Udara 9 AM (hPa)", 1000.0, 1035.0, 1015.0)
    min_t_in = st.slider("Min Temp", 1.8, 25.8, 15.0)
    max_t_in = st.slider("Max Temp", 9.1, 40.1, 25.0)
    rt_in = st.radio("Apakah Hari Ini Hujan?", ["No", "Yes"])

# --- 4. PREDIKSI ---
if st.button("Prediksi Sekarang"):
    # Gunakan spinner agar user tahu proses sedang berjalan
    with st.spinner('Menganalisis data...'):
        inputs = {
            'date': date_in, 'location': loc_in, 'humidity_3pm': hum_in,
            'rainfall': rain_in, 'sunshine': sun_in, 'wind_gust_speed': wind_s_in,
            'wind_gust_dir': wind_d_in, 'pressure_9am': press_in,
            'min_temp': min_t_in, 'max_temp': max_t_in, 'rain_today': rt_in
        }
        
        # 1. Pastikan nama variabel konsisten: final_df
        final_df = transform_user_input(inputs, assets)

        # 2. SEMUA KODE DI BAWAH INI HARUS MASUK INDENTASI (MENJOROK KE DALAM)
        # [cite_start]Menampilkan jumlah kolom untuk verifikasi [cite: 1214, 1216]
        st.write("Jumlah Kolom Input:", final_df.shape[1]) 
        
        if final_df.shape[1] != 70:
            [cite_start]st.warning(f"PERINGATAN: Jumlah kolom ({final_df.shape[1]}) tidak sesuai dengan model (70)!") [cite: 1216]

        # Cek apakah ada nilai NaN
        if final_df.isnull().values.any():
            st.warning("PERINGATAN: Ada nilai kosong (NaN) dalam data input!")
            st.write(final_df.columns[final_df.isna().any()].tolist())

        # 3. Eksekusi Prediksi
        res = model.predict(final_df)[0]
        prob = model.predict_proba(final_df)[0][1]

        # 4. Menampilkan Hasil di dalam blok tombol
        if res == 1:
            st.error(f"⚠️ Besok diprediksi HUJAN (Probabilitas: {prob:.1%})")
        else:
            st.success(f"☀️ Besok diprediksi CERAH (Probabilitas Hujan: {prob:.1%})")
