import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ASSETS ---
# Ubah jika file diletakkan langsung di root (tanpa folder models)
model = joblib.load('final_xgb_model.joblib')
assets = joblib.load('preprocessing_assets.joblib')

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

def transform_input(data_dict, assets):
    """
    Mengubah input UI menjadi DataFrame 70 kolom yang siap diprediksi.
    """
    # 1. Buat DataFrame kosong dengan 70 kolom (semua nol) sesuai blueprint
    input_df = pd.DataFrame(0, index=[0], columns=assets['feature_columns'])
    
    # 2. Isi Fitur Numerik Utama
    input_df['Rainfall'] = np.clip(data_dict['rainfall'], 0, 37.40) # Winsorizing [cite: 925]
    input_df['Humidity3pm'] = data_dict['humidity_3pm']
    input_df['Sunshine'] = data_dict['sunshine']
    input_df['WindGustSpeed'] = np.clip(data_dict['wind_gust_speed'], 15.0, 81.0) # [cite: 943]
    input_df['Pressure9am'] = data_dict['pressure_9am']
    input_df['MinTemp'] = data_dict['min_temp']
    input_df['MaxTemp'] = data_dict['max_temp']
    
    # 3. Isi Fitur Waktu [cite: 536-540]
    input_df['Year'] = data_dict['date'].year
    input_df['Month'] = data_dict['date'].month
    input_df['Day'] = data_dict['date'].day
    
    # 4. Mapping Kategorikal Biner & Wind [cite: 556, 615]
    input_df['RainToday'] = assets['rain_mapping'][data_dict['rain_today']]
    input_df['WindGustDir_Encoded'] = assets['wind_mapping'][data_dict['wind_gust_dir']]
    # Untuk WindDir9am & 3pm, kita bisa mengisi dengan modus data asli jika tidak diminta di UI [cite: 729]
    input_df['WindDir9am_Encoded'] = assets['imputation_values'].get('WindDir9am_Encoded', 0)
    input_df['WindDir3pm_Encoded'] = assets['imputation_values'].get('WindDir3pm_Encoded', 6)

    # 5. Aktifkan One-Hot Encoding Lokasi [cite: 1210, 1217]
    loc_column = f"Location_{data_dict['location']}"
    if loc_column in input_df.columns:
        input_df[loc_column] = 1
        
    return input_df

if st.button("Analisis Cuaca"):
    # Kumpulkan data dari widget UI
    user_data = {
        'rainfall': rainfall, 'humidity_3pm': humidity_3pm, 
        'sunshine': sunshine, 'wind_gust_speed': wind_gust_speed,
        'pressure_9am': pressure_9am, 'min_temp': min_temp,
        'max_temp': max_temp, 'location': location, 
        'wind_gust_dir': wind_dir, 'rain_today': rain_today,
        'date': date_input
    }
    
    # Transformasi
    final_input = transform_input(user_data, assets)
    
    # Prediksi [cite: 1256]
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1] # Probabilitas hujan
    
    # Tampilan Hasil
    if prediction == 1:
        st.error(f"⚠️ Prediksi: BESOK HUJAN (Kemungkinan: {probability:.1%})")
    else:
        st.success(f"☀️ Prediksi: BESOK CERAH (Kemungkinan Hujan: {probability:.1%})")import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ASSETS ---
# Tambahkan dekorator cache agar model hanya dimuat SATU KALI saat aplikasi pertama dibuka
@st.cache_resource
def load_assets():
    model = joblib.load('final_xgb_model.joblib')
    assets = joblib.load('preprocessing_assets.joblib')
    return model, assets

# Panggil fungsi
model, assets = load_assets()

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
    with st.spinner('Menghitung prediksi...'): # Spinner akan hilang otomatis setelah blok selesai
        user_data = { ... } # Kumpulkan data
        final_input = transform_input(user_data, assets)
        
        # Eksekusi prediksi
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]
        
    # Hasil ditampilkan di luar spinner
    if prediction == 1:
        st.error(f"⚠️ Prediksi: BESOK HUJAN ({probability:.1%})")
    else:
        st.success(f"☀️ Prediksi: BESOK CERAH ({probability:.1%})")


