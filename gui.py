import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_cropper import st_cropper

# Konfigurasi Halaman agar lebih lega
st.set_page_config(page_title="Image Processor", layout="wide")

st.title("Image Cropper")

# Sidebar untuk Input
uploaded_file = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # --- BAGIAN ATAS: AREA SELEKSI CROP ---
    st.subheader("1. Pilih Area yang Ingin Diambil")
    # Menggunakan kolom tengah agar gambar asli tidak terlalu besar menutupi layar
    _, col_crop_main = st.columns([1, 10])
    with col_crop_main:
        cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.divider()

    # --- BAGIAN BAWAH: PREVIEW DAN HISTOGRAM BERDAMPINGAN ---
    st.subheader("2. Hasil Analisis Area Terpilih")
    col_preview, col_hist = st.columns(2)

    # Konversi hasil crop ke Array untuk diolah
    cropped_array = np.array(cropped_img)

    with col_preview:
        st.write("**Preview Hasil Crop:**")
        st.image(cropped_img, use_container_width=True)

    with col_hist:
        st.write("**Histogram Intensitas:**")
        fig, ax = plt.subplots()
        
        # Logika Histogram
        if len(cropped_array.shape) == 3:  # Gambar Berwarna (RGB)
            colors = ('r', 'g', 'b')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([cropped_array], [i], None, [256], [0, 256])
                ax.plot(hist, color=col, linewidth=1.5)
        else:  # Gambar Grayscale
            hist = cv2.calcHist([cropped_array], [0], None, [256], [0, 256])
            ax.plot(hist, color='black', linewidth=1.5)
            
        ax.set_xlim([0, 256])
        ax.set_xlabel("Nilai Pixel")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    # --- BAGIAN INTEGRASI MODEL (DI SINI TEMPAT MODIFIKASI) ---
    st.divider()
    st.header("Integrasi Model Khusus")
    
    if st.button("Jalankan Prediksi Model"):
        with st.spinner('Model sedang memproses data...'):
            
            # ======================================================
            # TEMPAT MODIFIKASI MULAI
            # Di sini Anda bisa memasukkan preprocessing seperti:
            # img_input = cv2.resize(cropped_array, (lebar, tinggi))
            # result = model.predict(img_input)
            
            hasil_dummy = "Data siap diumpankan ke model AI. Area yang diproses adalah hasil crop di atas."
            
            # TEMPAT MODIFIKASI SELESAI
            # ======================================================

            st.success("Proses Berhasil!")
            st.info(f"Output Model: {hasil_dummy}")

else:
    st.info("Gunakan menu di samping kiri untuk mengunggah gambar yang ingin dianalisis.")