# modules/visualizer.py
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.figure_factory as ff
from modules import utils

def plot_histogram(image_array):
    fig, ax = plt.subplots(figsize=(5, 3))
    hist = cv2.calcHist([image_array], [0], None, [256], [0, 256])
    ax.plot(hist, color='black')
    ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.3)
    return fig

def plot_3d_surface(image_array):
    max_dim = 100
    h, w = image_array.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_array = cv2.resize(image_array, (int(w * scale), int(h * scale)))
    y = np.arange(image_array.shape[0])
    x = np.arange(image_array.shape[1])
    fig = go.Figure(data=[go.Surface(z=image_array, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(zaxis=dict(title='Z')))
    return fig

def plot_contour(image_array):
    fig = go.Figure(data=go.Contour(z=image_array, colorscale='Cividis'))
    return fig

def plot_gravity_flux(image_array):
    # Downsample agar tidak berat
    step = 10
    h, w = image_array.shape
    E, gx, gy = utils.compute_energy_field(image_array)

    # Kita balik gradiennya agar panah menunjuk ke arah 'gravitas' (low energy)
    # Sesuai teori: panah menunjuk ke arah aliran penurunan energi
    Y, X = np.mgrid[0:h:step, 0:w:step]
    U = -gx[::step, ::step]
    V = -gy[::step, ::step]

    fig = ff.create_quiver(X, Y, U, V, scale=0.1, arrow_scale=.3, name='Gravity Flux')
    fig.update_layout(title="Inward Flux Field (Vector Projection)", xaxis_title="X", yaxis_title="Y")
    return fig