# modules/metrics.py
import numpy as np
import modules.utils as utils
import numpy as np

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

def calc_lung_validation(image_array):
    """Contoh logika kompleks untuk validasi jaringan."""
    mean_val = np.mean(image_array)
    # Misal: Jaringan paru sehat biasanya berada di range tertentu
    status = "Potential Lung Tissue" if 50 < mean_val < 150 else "Check Density (Possible Bone/Fluid)"
    return {"Analysis Status": status}