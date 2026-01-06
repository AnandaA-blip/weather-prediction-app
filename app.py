import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('final_xgb_model.joblib')
    assets = joblib.load('preprocessing_assets.joblib')
    # Pastikan RainTomorrow tidak ikut sebagai fitur input
    if 'RainTomorrow' in assets['feature_columns']:
        assets['feature_columns'].remove('RainTomorrow')
    return model, assets

model, assets = load_assets()

# --- 2. FUNGSI TRANSFORMASI ---
def transform_user_input(data, assets):
    # Membuat baris data 0 sesuai jumlah fitur training (69 kolom)
    input_df = pd.DataFrame(0, index=[0], columns=assets['feature_columns'])
    
    # Mapping nilai numerik dengan clipping (Winsorizing) [cite: 923-925, 943]
    input_df['Rainfall'] = np.clip(data['rainfall'], 0, 37.40)
    input_df['Humidity3pm'] = data['humidity_3pm']
    input_df['WindGustSpeed'] = np.clip(data['wind_gust_speed'], 15.0, 81.0)
    input_df['Sunshine'] = data['sunshine']
    input_df['Pressure9am'] = data['pressure_9am']
    input_df['MinTemp'] = data['min_temp']
    input_df['MaxTemp'] = data['max_temp']
    
    # Fitur Waktu [cite: 536-540]
    input_df['Year'] = data['date'].year
    input_df['Month'] = data['date'].month
    input_df['Day'] = data['date'].day
    
    # Encoding Kategorikal [cite: 556, 615]
    input_df['RainToday'] = assets['rain_mapping'].get(data['rain_today'], 0)
    # Gunakan nama kolom yang sesuai dengan notebook (cek spasi vs underscore)
    wind_col = 'WindGustDir_Encoded' if 'WindGustDir_Encoded' in input_df.columns else 'WindGustDir Encoded'
    if wind_col in input_df.columns:
        input_df[wind_col] = assets['wind_mapping'].get(data['wind_gust_dir'], 12)

    # One-Hot Encoding Lokasi [cite: 1209-1215]
    loc_col = f"Location_{data['location']}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1
        
    return input_df

# --- 3. UI SIDEBAR ---
st.title("Aplikasi Prediksi Hujan Australia")
with st.sidebar:
    st.header("Input Data")
    date_in = st.date_input("Tanggal", datetime.now())
    loc_in = st.selectbox("Lokasi", assets['locations'])
    hum_in = st.slider("Kelembaban 3PM (%)", 0.0, 100.0, 50.0)
    rain_in = st.number_input("Curah Hujan (mm)", 0.0, 37.4, 0.0)
    sun_in = st.slider("Sinar Matahari (jam)", 0.0, 15.0, 7.0)
    wind_s_in = st.slider("Kecepatan Angin (km/j)", 15.0, 81.0, 40.0)
    wind_d_in = st.selectbox("Arah Angin", list(assets['wind_mapping'].keys()))
    press_in = st.number_input("Tekanan 9AM (hPa)", 1000.0, 1035.0, 1015.0)
    min_t_in = st.slider("Suhu Min", 1.8, 25.8, 15.0)
    max_t_in = st.slider("Suhu Max", 9.1, 40.1, 25.0)
    rt_in = st.radio("Hari Ini Hujan?", ["No", "Yes"])

# --- 4. PREDIKSI ---
if st.button("Prediksi Sekarang"):
    with st.spinner('Menganalisis...'):
        inputs = {
            'date': date_in, 'location': loc_in, 'humidity_3pm': hum_in,
            'rainfall': rain_in, 'sunshine': sun_in, 'wind_gust_speed': wind_s_in,
            'wind_gust_dir': wind_d_in, 'pressure_9am': press_in,
            'min_temp': min_t_in, 'max_temp': max_t_in, 'rain_today': rt_in
        }
        
        final_df = transform_user_input(inputs, assets)
        
        # Prediksi
        res = model.predict(final_df)[0]
        prob = model.predict_proba(final_df)[0][1]

        if res == 1:
            st.error(f"⚠️ Besok diprediksi HUJAN ({prob:.1%})")
        else:
            st.success(f"☀️ Besok diprediksi CERAH ({prob:.1%})")
