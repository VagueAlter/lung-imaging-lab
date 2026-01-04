# modules/visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def plot_basin_attraction(image_array):
    # 1. Pre-process: Smooth the image to avoid "micro-sinks" (Noise)
    smoothed = cv2.GaussianBlur(image_array.astype(np.float32), (5, 5), 0)
    
    # 2. Calculate Gradients
    dy, dx = np.gradient(smoothed)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # 3. Simulate Flow Accumulation (Simplified)
    # We'll identify "Sinks" by finding where gradient magnitude is near zero
    # and the Laplacian (curvature) is positive (a bowl shape).
    laplacian = cv2.Laplacian(smoothed, cv2.CV_32F)
    
    # Create an 'Attraction Map'
    # High attraction = Low Gradient Magnitude + High Positive Curvature
    attraction_map = np.where((magnitude < np.mean(magnitude)), laplacian, 0)
    attraction_map = np.clip(attraction_map, 0, None) # Only keep "bowls"
    
    # Normalize for visualization
    if attraction_map.max() > 0:
        attraction_map = (attraction_map / attraction_map.max()) * 100

    # 4. Plotly 3D Visualization
    y = np.arange(image_array.shape[0])
    x = np.arange(image_array.shape[1])
    
    fig = go.Figure(data=[go.Surface(
        z=image_array, 
        x=x, y=y,
        surfacecolor=attraction_map, # Color represents 'Attractiveness'
        colorscale='Hot',
        colorbar=dict(title="Attraction Intensity")
    )])
    
    fig.update_layout(
        title="Basins of Attraction (Heatmap on Surface)",
        scene=dict(zaxis=dict(title='Elevation (Z)')),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def plot_basin_attraction_sobel(image_array):
    # 1. Normalize and Smooth
    # Normalizing to 0-1 prevents floating point underflow on small slopes
    norm_img = cv2.normalize(image_array.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
    
    # 2. Use a Large Sobel Kernel (ksize=7) 
    # This 'bridges' the gap over flat pixels by looking at a 7x7 neighborhood
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=7)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=7)
    
    # 3. Calculate 'Attraction Force' (Gradient Magnitude)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. Visualization
    y, x = np.indices(image_array.shape)
    fig = go.Figure(data=[go.Surface(
        z=image_array, 
        x=x[0], y=y[:,0],
        surfacecolor=magnitude, # Colors show where slope change is most 'active'
        colorscale='Plasma',
        colorbar=dict(title="Slope Intensity")
    )])
    
    fig.update_layout(title="Basin Attraction: Sobel Gradient (Slope Energy)",
                      scene=dict(zaxis=dict(title='Z')))
    return fig

def plot_basin_attraction_dist_transform(image_array, threshold_factor=0.5):
    # 1. Create a mask of 'High Ground' (the walls of your basin)
    # We use a threshold based on the mean or a specific value
    threshold_val = np.mean(image_array) + (np.std(image_array) * threshold_factor)
    _, binary_mask = cv2.threshold(image_array.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
    
    # 2. Distance Transform: Calculate distance from the 'High Ground'
    # This creates a perfect slope toward the deepest/most isolated point
    dist_map = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)
    
    # 3. Visualization
    y, x = np.indices(image_array.shape)
    fig = go.Figure(data=[go.Surface(
        z=image_array, 
        x=x[0], y=y[:,0],
        surfacecolor=dist_map, # Colors show distance from walls
        colorscale='Twilight',
        colorbar=dict(title="Distance from Rim")
    )])
    
    fig.update_layout(title="Basin Attraction: Distance Transform (Geometric Center)",
                      scene=dict(zaxis=dict(title='Z')))
    return fig

def plot_rib_signal(image_array):
    score, s_i, peaks = utils.calculate_rib_periodicity(image_array)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=s_i, mode='lines', name='Gradient Projection'))
    fig.add_trace(go.Scatter(x=peaks, y=s_i[peaks], mode='markers', 
                             name='Detected Peaks', marker=dict(color='red', size=10)))
    
    fig.update_layout(title=f"Rib Signal (Score: {score:.2f})",
                      xaxis_title="Y-pixel (Vertical)",
                      yaxis_title="Sum of Positive Gradient")
    return fig

def plot_3d_autocorr_surface(image_array):
    autocorr = utils.calculate_3d_autocorr(image_array)
    
    # Visualisasi sebagai Surface Plot
    fig = go.Figure(data=[go.Surface(z=autocorr)])
    
    fig.update_layout(
        title='3D Autocorrelation Map',
        scene=dict(
            xaxis_title='X Lag',
            yaxis_title='Y Lag',
            zaxis_title='Correlation'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig