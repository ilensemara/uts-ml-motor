import streamlit as st
import pandas as pd
import joblib

# 1. Load Model dan Scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Tampilan Web
st.set_page_config(page_title="Dealer Motor Bekas AI", page_icon="🏍️")
st.title("🏍️ Sistem Penentuan Grade Motor")
st.write("Masukkan data motor untuk mengetahui kategorinya secara otomatis.")

# 3. Input User
with st.container():
    harga = st.number_input("Harga Motor (dalam Rp, contoh: 15000 untuk 15jt):", min_value=0)
    odometer = st.number_input("Jarak Tempuh / Odometer (KM):", min_value=0)

# 4. Tombol Prediksi
if st.button("Cek Kategori Sekarang"):
    # Preprocessing input sesuai dengan saat training
    input_data = pd.DataFrame([[harga, odometer]], columns=['harga', 'odometer (km)'])
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    
    # Hasil Berdasarkan Cluster
    st.subheader("Hasil Analisis AI:")
    if prediction == 0:
        st.success("✨ Kategori: **Grade A (Premium Condition)**")
        st.write("Karakteristik: Harga tinggi dengan jarak tempuh rendah.")
    elif prediction == 1:
        st.info("✅ Kategori: **Grade B (Good Value)**")
        st.write("Karakteristik: Harga dan pemakaian dalam kondisi wajar.")
    else:
        st.warning("🛒 Kategori: **Grade C (Budget Friendly)**")
        st.write("Karakteristik: Harga ekonomis dengan jarak tempuh tinggi.")