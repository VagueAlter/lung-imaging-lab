# modules/utils.py
import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, filters
from skimage.segmentation import flood_fill

def remove_dark_background(img, tolerance=30):
    """
    Mengubah background gelap menjadi putih menggunakan teknik Region Growing.
    """
    # Buat copy agar gambar asli tidak rusak
    result = img.copy()
    
    # Ambil koordinat 4 pojok sebagai benih (seeds)
    # Ambil dimensi gambar
    h, w = img.shape

    # Daftar titik pojok: kiri-atas, kanan-atas, kiri-bawah, kanan-bawah
    seeds = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    
    for seed in seeds:
        # flood_fill akan mencari pixel yang tersambung dengan seed
        # yang nilainya berada dalam rentang toleransi
        result = flood_fill(result, seed, new_value=255, tolerance=tolerance)
        
    return result

def compute_energy_field(image_array, lmbda=0.5):
    """
    E = I + lambda * ||grad I||
    Sesuai Section 1 di LaTeX.
    """
    I = image_array.astype(float)
    grad_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # E = I + lambda * ||grad I||
    E = I + lmbda * grad_mag
    return E, grad_x, grad_y

def calculate_inward_flux(E, center_p, r, R):
    """
    Menghitung S_i (Basin Trapping Criterion) sesuai Section 4 & 5.
    """
    h, w = E.shape
    yi, xi = center_p

    # Buat grid koordinat q
    yy, xx = np.mgrid[:h, :w]

    # Hitung vektor (pi - q)
    vec_x = xi - xx
    vec_y = yi - yy
    dist = np.sqrt(vec_x ** 2 + vec_y ** 2)

    # Definisi Neighborhood Ring Omega_i (Section 3)
    mask = (dist > r) & (dist <= R)

    if not np.any(mask): return 0.0

    # Hitung Gradien E
    grad_Ex = cv2.Sobel(E, cv2.CV_64F, 1, 0, ksize=3)
    grad_Ey = cv2.Sobel(E, cv2.CV_64F, 0, 1, ksize=3)

    # Normalisasi vektor unit (pi - q) / ||pi - q||
    unit_x = vec_x / (dist + 1e-5)
    unit_y = vec_y / (dist + 1e-5)

    # Proyeksi g(q; pi) = grad E . unit_vector (Section 4)
    g = (grad_Ex * unit_x) + (grad_Ey * unit_y)

    # S_i = Mean dari indicator function [g < 0] (Section 5)
    trapped_pixels = (g < 0) & mask
    S_i = np.sum(trapped_pixels) / np.sum(mask)

    return S_i

def calculate_gravity_flux(E, r_ratio=0.25):
    """
    Gravity-inspired: Mengecek apakah gradien sekitar mengarah ke pusat patch.
    Cocok untuk patch kecil yang berada di tengah paru.
    """
    h, w = E.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    
    # Vektor dari setiap titik ke pusat
    vx, vy = cx - xx, cy - yy
    dist = np.sqrt(vx**2 + vy**2) + 1e-5
    
    # Gradien Energy Field
    gx = cv2.Sobel(E, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(E, cv2.CV_64F, 0, 1, ksize=3)
    
    # Dot product: gradien . vektor_ke_pusat
    # Jika positif, berarti energi menurun (gradien negatif) menuju pusat
    dot = (gx * (vx/dist)) + (gy * (vy/dist))
    
    # Ambil rata-rata di ring tertentu
    mask = (dist > (min(h,w) * 0.1)) & (dist < (min(h,w) * r_ratio))
    gravity_score = np.mean(dot[mask]) if np.any(mask) else 0
    return gravity_score

def calculate_rib_periodicity(image_array, tau_min=10, tau_max=50):
    # 1. Hitung Gy = max(dI/dy, 0)
    # Kita gunakan Sobel untuk dI/dy
    gy = cv2.Sobel(image_array.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    gy_positive = np.maximum(gy, 0)
    
    # 2. Proyeksi 1D: s_i(y) = sum_x Gy(x,y)
    s_i = np.sum(gy_positive, axis=1)
    
    # 3. Peak Detection untuk S_peak
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(s_i, height=np.mean(s_i))
    
    if len(peaks) > 1:
        dist = np.diff(peaks)
        sigma_delta = np.std(dist)
        s_peak = len(peaks) / (sigma_delta + 1e-5)
    else:
        s_peak = 0.0

    # 4. Autokorelasi untuk S_auto
    # Normalisasi s_i (mean subtraction)
    s_hat = s_i - np.mean(s_i)
    n = len(s_hat)
    
    # Menghitung R_i(tau)
    r_tau_sum = 0
    # Pastikan tau tidak melebihi dimensi gambar
    t_max = min(tau_max, n - 1)
    
    denominator = np.sum(s_hat**2) + 1e-5
    for tau in range(tau_min, t_max + 1):
        # Shift and multiply
        numerator = np.sum(s_hat[tau:] * s_hat[:n-tau])
        r_tau_sum += (numerator / denominator)
        
    s_auto = r_tau_sum
    
    # S_total = S_peak * S_auto
    return s_peak * s_auto, s_i, peaks

def calculate_3d_autocorr(image_array):
    """
    Menghitung normalized 2D Autocorrelation sebagai representasi 
    pengulangan pola spasial (3D Surface Correlation).
    """
    # 1. Normalisasi sinyal (zero-mean)
    img_float = image_array.astype(float)
    img_norm = img_float - np.mean(img_float)
    
    # 2. FFT untuk menghitung korelasi (Teorema Wiener-Khinchin)
    fft_img = np.fft.fft2(img_norm)
    pow_spec = np.abs(fft_img)**2
    autocorr = np.fft.ifft2(pow_spec).real
    
    # 3. Shift agar lag (0,0) berada di tengah
    autocorr = np.fft.fftshift(autocorr)
    
    # 4. Normalisasi agar nilai max = 1.0 (at zero lag)
    denominator = (np.sum(img_norm**2) + 1e-5)
    autocorr = autocorr / denominator
    
    return autocorr

def extract_3d_periodicity_features(autocorr):
    """Ekstraksi skor dari permukaan autokorelasi."""
    h, w = autocorr.shape
    center_y, center_x = h // 2, w // 2
    
    # Kita ambil "ring" di sekitar pusat untuk mencari peak kedua (periodisitas)
    # Skor tinggi jika ada puncak-puncak sekunder selain di titik (0,0)
    total_strength = np.sum(np.abs(autocorr)) - autocorr[center_y, center_x]
    
    return total_strength

def calculate_boost_factors(cropped_array, full_image_shape, crop_coordinates):
    h_full, w_full = full_image_shape
    x, y, w, h = crop_coordinates
    
    # 1. Centroid Factor (0.0 = Pinggir, 1.0 = Tengah Vertikal)
    mid_x_full = w_full / 2
    centroid_x = x + (w / 2)
    distance_to_mid = abs(centroid_x - mid_x_full)
    
    # Normalisasi: Jarak 0 di tengah jadi 1.0, jarak maksimal di pinggir jadi 0.0
    distance_score = max(0.0, 1.0 - (distance_to_mid / mid_x_full))
    
    # 2. Scale Factor (0.0 = Sangat Kecil, 1.0 = Seluruh Gambar)
    area_crop = w * h
    area_full = w_full * h_full
    scale_score = area_crop / area_full # Sudah pasti antara 0 dan 1
    
    return distance_score, scale_score