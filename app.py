import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Model dan Pendukungnya
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 2. Judul Dashboard
st.set_page_config(page_title="HR Productivity Predictor", layout="wide")
st.title("🧵 Dashboard Prediksi Produktivitas Karyawan Garmen")
st.write("Gunakan panel di kiri untuk memasukkan data operasional harian.")

# 3. Sidebar untuk Input Data
st.sidebar.header("Input Data Operasional")

def user_input_features():
    # Fitur Numerik Utama
    targeted_productivity = st.sidebar.slider("Target Produktivitas", 0.0, 1.0, 0.8)
    smv = st.sidebar.number_input("Standard Minute Value (SMV)", 0.0, 60.0, 20.0)
    wip = st.sidebar.number_input("Work In Progress (WIP)", 0, 10000, 100)
    over_time = st.sidebar.number_input("Overtime (Menit)", 0, 15000, 2000)
    incentive = st.sidebar.number_input("Incentive (Bonus)", 0, 1000, 50)
    idle_time = st.sidebar.number_input("Idle Time", 0.0, 10.0, 0.0)
    idle_men = st.sidebar.number_input("Idle Men", 0, 50, 0)
    no_of_style_change = st.sidebar.number_input("Jumlah Perubahan Gaya", 0, 10, 0)
    no_of_workers = st.sidebar.number_input("Jumlah Pekerja", 1, 100, 50)
    
    # Fitur Kategorikal (Sederhana saja untuk demo)
    department = st.sidebar.selectbox("Departemen", ["sewing", "finishing"])
    
    # Membuat DataFrame dari input
    data = {
        'targeted_productivity': targeted_productivity,
        'smv': smv,
        'wip': wip,
        'over_time': over_time,
        'incentive': incentive,
        'idle_time': idle_time,
        'idle_men': idle_men,
        'no_of_style_change': no_of_style_change,
        'no_of_workers': no_of_workers,
        'department_sewing': 1 if department == 'sewing' else 0,
        # Catatan: Kolom lain seperti quarter dan day diset 0 secara default untuk kemudahan dashboard
    }
    
    # Pastikan semua kolom yang dibutuhkan model ada di sini
    features = pd.DataFrame([data])
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
            
    # Pastikan urutan kolom sesuai saat training
    features = features[feature_columns]
    return features

input_df = user_input_features()

# 4. Tampilkan Input Manajer HR
st.subheader("Data yang dimasukkan:")
st.write(input_df)

# 5. Prediksi
if st.button("Prediksi Sekarang"):
    # Scaling input
    input_scaled = scaler.transform(input_df)
    
    # Jalankan Model
    prediction = model.predict(input_scaled)
    
    # Tampilkan Hasil
    st.header(f"Hasil Prediksi: {prediction[0]:.2%}")
    
    target = input_df['targeted_productivity'].values[0]
    if prediction[0] >= target:
        st.success("✅ Prediksi: Target Tercapai!")
    else:
        st.error("⚠️ Prediksi: Target Gagal Tercapai.")