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
        
        # Nama variabel adalah final_df
        final_df = transform_user_input(inputs, assets)

        # --- DEBUGGING (Menggunakan nama variabel yang benar: final_df) ---
        st.write("Jumlah Kolom Input:", final_df.shape[1]) [cite: 1214, 1216]
        
        if final_df.shape[1] != 70:
            st.warning(f"PERINGATAN: Jumlah kolom ({final_df.shape[1]}) tidak sesuai (Harusnya 70)!") [cite: 1216]

        # Cek apakah ada nilai NaN
        if final_df.isnull().values.any():
            st.warning("PERINGATAN: Ada nilai kosong (NaN) dalam data input!")
            st.write(final_df.columns[final_df.isna().any()].tolist())

        # Eksekusi Prediksi (Pastikan berada di dalam blok indentasi button)
        res = model.predict(final_df)[0] [cite: 1256]
        prob = model.predict_proba(final_df)[0][1]

        # Menampilkan Hasil
        if res == 1:
            st.error(f"⚠️ Besok diprediksi HUJAN (Probabilitas: {prob:.1%})") [cite: 1245]
        else:
            st.success(f"☀️ Besok diprediksi CERAH (Probabilitas Hujan: {prob:.1%})") [cite: 1245]
