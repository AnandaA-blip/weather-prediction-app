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
def transform_user_input(data, assets, model):
    # Mengambil daftar fitur ASLI yang diinginkan oleh model XGBoost
    model_features = model.get_booster().feature_names
    
    # Membuat DataFrame kosong dengan kolom yang sama persis dengan model
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Mapping nilai dari UI ke kolom
    mapping_values = {
        'MinTemp': np.clip(data['min_temp'], 1.80, 25.80), # [cite: 3510]
        'MaxTemp': np.clip(data['max_temp'], 9.10, 40.10), # [cite: 3510]
        'Rainfall': np.clip(data['rainfall'], 0, 37.40),  # [cite: 3510]
        'Sunshine': data['sunshine'],
        'WindGustSpeed': np.clip(data['wind_gust_speed'], 15.0, 81.0), # [cite: 3510]
        'Humidity9am': assets['imputation_values'].get('Humidity9am', 70.0),
        'Humidity3pm': data['humidity_3pm'],
        'Pressure9am': np.clip(data['pressure_9am'], 1000.20, 1034.00), # [cite: 3510]
        'Year': data['date'].year,
        'Month': data['date'].month,
        'Day': data['date'].day,
        'RainToday': assets['rain_mapping'].get(data['rain_today'], 0),
        'WindGustDir_Encoded': assets['wind_mapping'].get(data['wind_gust_dir'], 12)
    }

    # Mengisi nilai yang ada. Jika model minta 'Temp9am' yang dihapus, 
    # kita isi dengan nilai median/imputasi agar tidak error.
    for col in model_features:
        if col in mapping_values:
            input_df[col] = mapping_values[col]
        else:
            # Gunakan nilai imputasi (median) untuk fitur redundan/tidak ada di UI
            input_df[col] = assets['imputation_values'].get(col, 0)

    # One-Hot Encoding Lokasi
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
    with st.spinner('Menganalisis data...'):
        inputs = {
            'date': date_in, 'location': loc_in, 'humidity_3pm': hum_in,
            'rainfall': rain_in, 'sunshine': sun_in, 'wind_gust_speed': wind_s_in,
            'wind_gust_dir': wind_d_in, 'pressure_9am': press_in,
            'min_temp': min_t_in, 'max_temp': max_t_in, 'rain_today': rt_in
        }
        
        # Tambahkan argumen 'model' ke dalam fungsi
        final_df = transform_user_input(inputs, assets, model)

        # DEBUGGING: Sekarang menggunakan final_df (bukan final_input)
        st.write(f"Model mengharapkan {len(model.get_booster().feature_names)} fitur.")
        st.write(f"Data input memiliki {final_df.shape[1]} fitur.")

        # Eksekusi Prediksi
        res = model.predict(final_df)[0]
        prob = model.predict_proba(final_df)[0][1]

        if res == 1:
            st.error(f"⚠️ Besok HUJAN ({prob:.1%})")
        else:
            st.success(f"☀️ Besok CERAH ({prob:.1%})")


