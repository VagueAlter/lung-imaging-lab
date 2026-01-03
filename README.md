# 🫁 Lung Imaging Lab

**Lung Imaging Lab** is a modular research sandbox for the mathematical exploration of medical image patches.  
Built with **Streamlit**, this dashboard enables researchers to prototype, test, and validate diverse formulations for lung tissue characterization and patch expansion in Chest X-Ray (CXR) images.

Rather than relying on black-box AI models, this project emphasizes **transparent, physics-informed, and geometry-aware formulations** operating on localized image domains.

---

## 🎯 Project Philosophy

This repository is intentionally designed as a **laboratory**, not a single-method solution.

It aims to answer questions such as:
- *What mathematical formulation best captures “lung-likeness” under different contexts?*
- *How does local structure change as a patch expands spatially?*
- *Which scalar fields or geometric descriptors are stable across patients and acquisition conditions?*

Methods may evolve — the **framework remains**.

---

## 🚀 Key Features

- **Modular Metrics Framework**  
  Any function prefixed with `calc_` inside `modules/metrics.py` is automatically discovered and rendered.

- **Dynamic Multi-Modal Visualization**  
  Real-time rendering of:
  - Intensity histograms  
  - 3D surface representations  
  - Vector (flux) fields  
  - Contour maps (optional)

- **Physics-Informed Preprocessing**  
  Gaussian smoothing and contrast stretching stabilize gradient-based and energy-based formulations.

- **Interactive Patch Exploration**  
  Real-time cropping allows immediate feedback on how patch position and size affect derived metrics.

---

## 🧪 Theoretical Foundation: Gravity-Inspired Basin Detection

One of the currently implemented formulations models lung regions as **low-energy basins** within an image-derived scalar field.

### 1. Energy Field Definition

An energy landscape is defined as:

\[
E(x, y) = I(x, y) + \lambda \left\lVert \nabla I(x, y) \right\rVert
\]

where:
- \( I(x, y) \) is the image intensity  
- \( \nabla I \) is the spatial gradient  
- \( \lambda \) controls edge sensitivity

---

### 2. Inward Flux (Basin Trapping Criterion)

For a patch centered at \( p_i \), its surrounding neighborhood \( \Omega_i \) is evaluated using a directional flux test:

\[
g(q; p_i) = \nabla E(q) \cdot \frac{p_i - q}{\lVert p_i - q \rVert}
\]

The **basin trapping score** is defined as:

\[
S_i = \frac{1}{|\Omega_i|} \sum_{q \in \Omega_i} \mathbb{I}\left[g(q; p_i) < 0\right]
\]

A patch is considered *lung-like* if:

\[
S_i \ge \tau
\]

where \( \tau \) is a tunable threshold.

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit dashboard
├── modules/
│   ├── visualizer.py       # Rendering logic (3D, contour, vector fields)
│   ├── metrics.py          # Patch-level formulations and descriptors
│   └── utils.py            # Core mathematical & physics primitives
├── research/
│   ├── notes.tex           # Exploratory derivations & formulation notes
│   ├── gravity_model.tex   # Basin / flux-based theory writeup
│   └── figures/            # Exported plots and diagrams for papers
├── requirements.txt        # Dependencies (OpenCV, Plotly, Streamlit, etc.)
└── README.md
