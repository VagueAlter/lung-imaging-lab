# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_cropper import st_cropper
import inspect
import modules.visualizer as viz
import modules.metrics as metrics 

st.set_page_config(page_title="Modular Lung Analyzer", layout="wide")

ACTIVE_PLOTS = [
    ("📊 Histogram", viz.plot_histogram),
    ("🧊 3D Surface", viz.plot_3d_surface),
    # ("🗺️ Contour", viz.plot_contour),
    ("🏹 Gravity Flux", viz.plot_gravity_flux)
]

metric_functions = [f for name, f in inspect.getmembers(metrics, inspect.isfunction) if name.startswith('calc_')]

st.title("🖼️ Modular Image Analysis")

uploaded_file = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_raw = np.array(Image.open(uploaded_file).convert('L'))
    img_smooth = cv2.GaussianBlur(img_raw, (5, 5), 0)
    p_min, p_max = np.min(img_smooth), np.max(img_smooth)
    img_stretched = ((img_smooth - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    img_final = Image.fromarray(img_stretched)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("1. Area Seleksi")
        cropped_img = st_cropper(img_final, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        cropped_array = np.array(cropped_img)

        st.divider()
        st.subheader("📊 Patch Metrics")

        for func in metric_functions:
            data = func(cropped_array)
            cols = st.columns(len(data))
            for i, (label, value) in enumerate(data.items()):
                cols[i].metric(label, value)

    with col_right:
        st.subheader("2. Visualisasi")
        tabs = st.tabs([item[0] for item in ACTIVE_PLOTS])
        for i, (title, func) in enumerate(ACTIVE_PLOTS):
            with tabs[i]:
                result = func(cropped_array)
                if hasattr(result, 'to_json'):
                    st.plotly_chart(result, use_container_width=True)
                else:
                    st.pyplot(result)
else:
    st.info("Silakan unggah gambar di sidebar.")