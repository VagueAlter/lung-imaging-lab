# modules/utils.py
import numpy as np
import cv2

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