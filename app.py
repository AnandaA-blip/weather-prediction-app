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
    # Buat dataframe dengan 0 berdasarkan kolom asli training
    input_df = pd.DataFrame(0, index=[0], columns=assets['feature_columns'])
    
    # Gunakan pemetaan yang sangat hati-hati terhadap nama kolom
    # Pastikan nama kolom di bawah ini sama persis dengan assets['feature_columns']
    mapping_values = {
        'Rainfall': np.clip(data['rainfall'], 0, 37.40),
        'Humidity3pm': data['humidity_3pm'],
        'WindGustSpeed': np.clip(data['wind_gust_speed'], 15.0, 81.0),
        'Sunshine': data['sunshine'],
        'Pressure9am': data['pressure_9am'],
        'MinTemp': data['min_temp'],
        'MaxTemp': data['max_temp'],
        'Year': data['date'].year,
        'Month': data['date'].month,
        'Day': data['date'].day,
        'RainToday': assets['rain_mapping'].get(data['rain_today'], 0),
        # CEK DISINI: Apakah di model Anda pakai spasi atau underscore?
        'WindGustDir_Encoded': assets['wind_mapping'].get(data['wind_gust_dir'], 12),
        'WindDir9am_Encoded': assets['imputation_values'].get('WindDir9am_Encoded', 0),
        'WindDir3pm_Encoded': assets['imputation_values'].get('WindDir3pm_Encoded', 6)
    }

    # Isi nilai ke input_df hanya jika kolomnya ada
    for col, val in mapping_values.items():
        if col in input_df.columns:
            input_df[col] = val

    # Lokasi (One-Hot Encoding)
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
        
        final_df = transform_user_input(inputs, assets)
# --- DEBUGGING (Tambahkan ini sementara) ---
st.write("Jumlah Kolom Input:", final_input.shape[1])
if final_input.shape[1] != 70:
    st.warning(f"PERINGATAN: Jumlah kolom ({final_input.shape[1]}) tidak sesuai dengan model (70)!")

# Cek apakah ada nilai NaN
if final_input.isnull().values.any():
    st.warning("PERINGATAN: Ada nilai kosong (NaN) dalam data input!")
    st.write(final_input.columns[final_input.isna().any()].tolist())
        # Eksekusi Prediksi [cite: 1256]
res = model.predict(final_df)[0]
prob = model.predict_proba(final_df)[0][1]

    # Menampilkan Hasil
if res == 1:
    st.error(f"⚠️ Besok diprediksi HUJAN (Probabilitas: {prob:.1%})")
else:
    st.success(f"☀️ Besok diprediksi CERAH (Probabilitas Hujan: {prob:.1%})")


