import numpy as np
import cv2
from scipy.signal import find_peaks

def get_periodicity_features(cropped_array):
    """Menghitung S_peak dan S_auto dari proyeksi gradien vertikal."""
    gy = cv2.Sobel(cropped_array.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    gy_pos = np.maximum(gy, 0)
    s_y = np.sum(gy_pos, axis=1) # Proyeksi 1D
    
    # 1. S_peak
    peaks, _ = find_peaks(s_y, height=np.mean(s_y))
    if len(peaks) > 1:
        dist_peaks = np.diff(peaks)
        s_peak = len(peaks) / (np.std(dist_peaks) + 1e-5)
    else:
        s_peak = 0.0
    s_peak_norm = max(0.0, min(1.0, s_peak / 10.0))

    # 2. S_auto
    s_hat = s_y - np.mean(s_y)
    autocorr_full = np.correlate(s_hat, s_hat, mode='full')
    autocorr_signal = autocorr_full[len(autocorr_full)//2:] 
    s_auto = np.sum(autocorr_signal[10:50]) / (np.sum(s_hat**2) + 1e-5)
    s_auto_norm = max(0.0, min(1.0, s_auto))
    
    return s_peak_norm, s_auto_norm

def get_spatial_context(box, full_shape):
    """Menghitung Distance Score dan Scale Score."""
    full_h, full_w = full_shape
    left, top, width, height = box['left'], box['top'], box['width'], box['height']
    
    # Distance Score
    mid_x_full = full_w / 2
    centroid_x = left + (width / 2)
    dist_score = max(0.0, 1.0 - (abs(centroid_x - mid_x_full) / mid_x_full))
    
    # Scale Score
    scale_score = (width * height) / (full_w * full_h)
    
    return dist_score, scale_score