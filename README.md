# GGR - Geostatistical Grid Realization

Modern geostatistics in Python - kriging and simulation for the browser and beyond.

## Features

- **Pure Python** - Runs anywhere, including JupyterLite in your browser
- **Simple & Powerful API** - Both class-based and functional interfaces
- **2D & 3D Support** - Flexible spatial dimensions
- **Comprehensive Diagnostics** - Kriging variance, efficiency, and more
- **Anisotropic Search** - Full 3D anisotropy with multiple angle conventions
- **GSLIB-Validated** - Tested against industry-standard GSLIB

## Quick Start

```python
import ggr
import numpy as np

# Your drill hole data
data = np.array([
    [x1, y1, z1, grade1],
    [x2, y2, z2, grade2],
    # ...
])

# Define variogram (or use stereogamma library)
variogram = {
    'model': 'spherical',
    'range': 500,
    'sill': 1.0,
    'nugget': 0.1
}

# Define estimation grid
grid = {
    'xmin': 0, 'xmax': 1000, 'nx': 100,
    'ymin': 0, 'ymax': 1000, 'ny': 100,
    'zmin': 0, 'zmax': 100, 'nz': 10
}

# Krige it!
result = ggr.krige(
    data,
    variogram=variogram,
    grid=grid,
    method='ordinary',
    search_radius=500
)
```

## Installation

### Standard Installation
```bash
pip install ggr
```

### For JupyterLite
```python
import micropip
await micropip.install('ggr')
```

### Development Installation
```bash
git clone https://github.com/yourusername/ggr.git
cd ggr
pip install -e ".[dev]"
```

## Methods Supported

### Kriging
- **Ordinary Kriging (OK)** - Unknown mean, estimated from data
- **Simple Kriging (SK)** - Known mean, provided by user

### Coming Soon
- Sequential Gaussian Simulation (SGS)
- Indicator Kriging
- Co-kriging

## API Overview

### Functional API (Quick & Easy)
```python
estimates = ggr.krige(data, variogram, grid, method='ordinary')
```

### Class API (More Control)
```python
krige = ggr.OrdinaryKrige(variogram=variogram, search_radius=500)
krige.fit(data)
estimates = krige.predict(grid)
```

### With Full Diagnostics
```python
result = ggr.krige(
    data, variogram, grid,
    return_variance=True,
    return_diagnostics=True
)

# Access outputs
estimates = result['estimates']
variance = result['variance']
n_samples = result['n_samples']
efficiency = result['kriging_efficiency']
```

## Anisotropic Search

```python
# Define anisotropic search ellipsoid
search = ggr.AnisotropicSearch(
    radius=[500, 300, 100],  # major, semi-major, minor
    dip_direction=45,  # degrees
    dip=30,
    rake=0,
    octants=True,  # enforce octant search
    max_points=50,
    min_points=4
)

result = ggr.krige(data, variogram, grid, search=search)
```

## Testing & Validation

GGR is validated against GSLIB outputs to ensure numerical correctness:

```bash
pytest tests/test_gslib_validation.py
```

## Design Philosophy

1. **Browser-First** - Works in JupyterLite without compromise
2. **Type-Flexible** - Accept pandas DataFrames or NumPy arrays
3. **Clean APIs** - Intuitive interfaces following Python best practices
4. **Well-Tested** - Comprehensive test suite with GSLIB validation
5. **Educational** - Clear code and documentation

## Dependencies

- `numpy` - Array operations
- `scipy` - Spatial indexing and linear algebra
- `pandas` - Data handling
- `matplotlib` (optional) - Visualization

All dependencies work in Pyodide/JupyterLite!

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built on the shoulders of giants:
- GSLIB by Deutsch & Journel
- Modern Python scientific stack
- Inspired by PyKrige, GSTools, and GeostatsPy

---

**GGR** - Bringing geostatistics to the browser
