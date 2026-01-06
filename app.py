import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ASSETS ---
model = joblib.load('models/final_xgb_model.joblib')
assets = joblib.load('models/preprocessing_assets.joblib')

st.title("Aplikasi Prediksi Hujan Australia")
st.write("Masukkan parameter cuaca di bawah untuk memprediksi hujan esok hari.")

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("Parameter Meteorologi")

# Input Numerik berdasarkan range Winsorizing 
humidity_3pm = st.sidebar.slider("Kelembaban Jam 3 Sore (%)", 0.0, 100.0, 50.0)
rainfall = st.sidebar.slider("Curah Hujan Hari Ini (mm)", 0.0, 37.4, 0.0)
sunshine = st.sidebar.slider("Jam Sinar Matahari", 0.0, 15.0, 8.0)
wind_gust_speed = st.sidebar.slider("Kecepatan Angin Maks (km/jam)", 15.0, 81.0, 40.0)
pressure_9am = st.sidebar.slider("Tekanan Udara 9 Pagi (hPa)", 1000.0, 1034.0, 1015.0)

# Input Kategorikal [cite: 1060, 615]
location = st.sidebar.selectbox("Lokasi Stasiun", assets['locations'])
wind_dir = st.sidebar.selectbox("Arah Angin Terkencang", list(assets['wind_mapping'].keys()))
rain_today = st.sidebar.selectbox("Apakah Hari Ini Hujan?", ["No", "Yes"])

# Input Tanggal (untuk Year, Month, Day) [cite: 535]
date_input = st.sidebar.date_input("Tanggal Observasi")

# --- 3. PREDICTION LOGIC ---
if st.button("Prediksi Cuaca"):
    # Di sini kita akan memasukkan logika transformasi data menjadi 70 kolom
    st.write("Sedang menghitung...")