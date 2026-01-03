# Lung Imaging Lab

**Lung Imaging Lab** is a modular research workstation designed for the mathematical exploration of medical image patches. 

Built with **Streamlit**, this dashboard provides a robust environment for researchers to prototype, visualize, and validate mathematical formulations applied to localized image domains in Chest X-Ray (CXR) images.


## Project Purpose

This repository is designed as a **formulation laboratory**. It is built to test easily on how different mathematical descriptors behave when applied to specific regions of a medical image.

The primary goal is to provide a sandbox where you can:
- **Experiment** with new formulas for tissue characterization.
- **Analyze** the geometric and statistical properties of image patches in real-time.
- **Develop** logic for patch expansion and anatomical fitting.

## Key Features

### 1. Automated Metric Discovery
The application uses a dynamic discovery system. Any mathematical function defined in `modules/metrics.py` (prefixed with `calc_`) is automatically detected and rendered as a live metric in the dashboard.

### 2. Multi-Modal Visualization Suite
Supports a variety of real-time rendering modes to inspect patch data from different perspectives:
- **Intensity Analysis**: Histograms and pixel distribution.
- **Surface Modeling**: 3D mesh representation of image intensity.
- **Field Visualization**: Vector and gradient field projections.
- **Topological Mapping**: Contour and isoline visualizations.

### 3. Interactive Exploration
Using a real-time cropping interface, users can move "probes" across the CXR to see instantly how local coordinates and image content affect the calculated scores and visual output.


## 📁 Project Structure

```text
├── app.py                # Main Streamlit dashboard and UI logic
├── modules/
│   ├── visualizer.py     # Rendering logic for all plots and charts
│   ├── metrics.py        # Implementation of patch-level formulas
│   └── utils.py          # Core mathematical and image primitives
├── research/             # Mathematical derivations and LaTeX notes
├── requirements.txt      # Project dependencies
└── README.md
```

#  Getting Started
Clone the repository:

```Bash

git clone https://github.com/VagueAlter/lung-imaging-lab
cd lung-imaging-lab
```

Install Dependencies:

```Bash

pip install -r requirements.txt
```

Launch the Lab:

```Bash
streamlit run app.py
```