# modules/metrics.py
import numpy as np
import modules.utils as utils
import numpy as np
import streamlit as st

def calc_basic_stats(image_array):
    """Menghitung statistik dasar intensitas."""
    mean_val = np.mean(image_array)
    std_val = np.std(image_array)
    return {
        "Mean Intensity": f"{mean_val:.2f}",
        "Standard Deviation": f"{std_val:.2f}",
        "Max Intensity": int(np.max(image_array)),
        "Min Intensity": int(np.min(image_array))
    }

def calc_geometric_info(image_array):
    """Menghitung informasi geometri patch."""
    h, w = image_array.shape
    aspect_ratio = w / h
    return {
        "Dimensions": f"{w}x{h} px",
        "Area": f"{w * h} px²",
        "Aspect Ratio": f"{aspect_ratio:.2f}"
    }

def calc_gravity_basin_score(cropped_array):
    """
    Menghitung skor probabilitas apakah patch berada di dalam basin paru.
    """
    # 1. Hitung Energy Field
    E, _, _ = utils.compute_energy_field(cropped_array)

    # 2. Tentukan pusat patch pi
    h, w = cropped_array.shape
    center_p = (h // 2, w // 2)

    # 3. Hitung Flux (S_i)
    # r = radius dalam (patch), R = context radius (luar patch)
    r = min(h, w) // 4
    R = max(h, w) // 2

    S_i = utils.calculate_inward_flux(E, center_p, r, R)

    # 4. Interpretasi sesuai Section 6
    status = "Lung Basin Detected" if S_i > 0.6 else "Anisotropic/Boundary"

    return {
        "Basin Score (S_i)": f"{S_i:.2f}",
        "Trapping Status": status
    }

def calc_rib_pattern_score(image_array):
    """Menghitung skor periodisitas rusuk berdasarkan research note."""
    # Kita panggil utility yang tadi dibuat
    score, _, _ = utils.calculate_rib_periodicity(image_array)
    
    return {
        "Rib Periodicity Score": f"{score:.4f}",
        "Pattern Strength": "High" if score > 1.0 else "Low/None"
    }

def calc_3d_spatial_periodicity(image_array):
    """Metrik berdasarkan Autokorelasi Permukaan (3D)."""
    autocorr = utils.calculate_3d_autocorr(image_array)
    score = utils.extract_3d_periodicity_features(autocorr)
    
    return {
        "3D Pattern Strength": f"{score:.2f}",
        "Spatial Coherence": "High" if score > 50 else "Diffuse"
    }

def calc_boosted_basin_score(cropped_array):
    # 1. Ambil skor basin dasar (asumsi rentang 0-1)
    basic_results = calc_gravity_basin_score(cropped_array)
    s_i = float(basic_results["Basin Score (S_i)"])
    s_i = max(0.0, min(1.0, s_i)) # Guard agar tetap 0-1

    # 2. Ambil faktor dari koordinat
    if 'last_crop_coords' in st.session_state:
        coords = st.session_state.last_crop_coords
        full_shape = st.session_state.full_img_shape
        dist_f, scale_f = utils.calculate_boost_factors(cropped_array, full_shape, coords)
    else:
        dist_f, scale_f = 0.5, 0.1 # Default

    # 3. Rumus Boosted (Multiplikatif agar Maksimal 1)
    # Kita beri sedikit 'base' agar tidak langsung nol jika di pinggir
    # Contoh: Boosted = S_i * (porsi_posisi + porsi_skala)
    # Atau lebih sederhana jika ingin S_i sebagai faktor utama:
    
    w1, w2, w3 = 0, 0.0, 1.0 # Total bobot = 1.0
    boosted_score = (s_i * w1) + (dist_f * w2) + (scale_f * w3)
    
    # Jika kamu tetap ingin S_i dikali booster tapi tetap maksimal 1:
    # boosted_score = s_i * ((dist_f * 0.7) + (scale_f * 0.3))

    return {
        "Base S_i": f"{s_i:.3f}",
        "Boosted Score": f"{boosted_score:.3f}",
        "Confidence": f"{boosted_score * 100:.1f}%"
    }