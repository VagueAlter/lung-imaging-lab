import streamlit as st
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import inspect
import cv2
import modules.visualizer as viz
import modules.metrics as metrics
import modules.engine as engine # Import engine baru

st.set_page_config(page_title="Modular Lung Analyzer", layout="wide")

WEIGHTS = {"peak": 0.35, "auto": 0.25, "scale": 0.20, "dist": 0.15, "basin": 0.05}

ACTIVE_PLOTS = [
    ("📊 Histogram", viz.plot_histogram),
    ("🧊 3D Surface", viz.plot_3d_surface),
    ("🩻 Ribs", viz.plot_rib_signal),
    ("Surface", viz.plot_3d_autocorr_surface)
]

metric_functions = [f for name, f in inspect.getmembers(metrics, inspect.isfunction) if name.startswith('calc_')]

st.title("🖼️ Modular Image Analysis")
uploaded_file = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # --- PREPROCESSING ---
    img_raw = np.array(Image.open(uploaded_file).convert('L'))
    img_smooth = cv2.GaussianBlur(img_raw, (5, 5), 0)
    p_min, p_max = np.min(img_smooth), np.max(img_smooth)
    img_stretched = ((img_smooth - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    img_final = Image.fromarray(img_stretched)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("1. Area Seleksi")
        box = st_cropper(img_final, realtime_update=True, box_color='#FF0000', aspect_ratio=None, return_type='box')
        
        # Crop Data
        cropped_array = img_stretched[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]

        # --- CALCULATIONS ---
        # 1. Periodicity Features
        s_peak_n, s_auto_n = engine.get_periodicity_features(cropped_array)
        # 2. Spatial Context
        dist_s, scale_s = engine.get_spatial_context(box, img_stretched.shape)
        # 3. Basin Score
        s_i = float(metrics.calc_gravity_basin_score(cropped_array)["Basin Score (S_i)"])

        # Final Score
        final_score = (s_i * WEIGHTS["basin"]) + (s_peak_n * WEIGHTS["peak"]) + \
                      (s_auto_n * WEIGHTS["auto"]) + (dist_s * WEIGHTS["dist"]) + (scale_s * WEIGHTS["scale"])

        # --- UI DISPLAY ---
        st.success(f"### 🎯 Final Research Score: {final_score:.4f}")
        st.progress(min(1.0, final_score))

        cols = st.columns(5)
        cols[0].metric("Basin", f"{s_i:.2f}", f"{WEIGHTS['basin']}")
        cols[1].metric("Peak D.", f"{s_peak_n:.2f}", f"{WEIGHTS['peak']}")
        cols[2].metric("Auto C.", f"{s_auto_n:.2f}", f"{WEIGHTS['auto']}")
        cols[3].metric("Dist", f"{dist_s:.2f}", f"{WEIGHTS['dist']}")
        cols[4].metric("Scale", f"{scale_s:.2f}", f"{WEIGHTS['scale']}")

        st.divider()
        st.subheader("📊 Patch Metrics")
        for func in metric_functions:
            data = func(cropped_array)
            m_cols = st.columns(len(data))
            for i, (label, value) in enumerate(data.items()):
                m_cols[i].metric(label, value)

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