# GGR Quick Start Example

This notebook demonstrates basic usage of GGR for kriging.

## Setup

```python
import numpy as np
import pandas as pd
import ggr

# Optional: for visualization
import matplotlib.pyplot as plt
```

## Generate Sample Data

```python
# Create synthetic drill hole data
np.random.seed(42)
n_samples = 50

x = np.random.rand(n_samples) * 1000
y = np.random.rand(n_samples) * 1000
z = np.random.rand(n_samples) * 200

# Create spatial structure with some noise
grade = (
    5.0 +                              # base grade
    2.0 * np.sin(x / 200) +           # E-W trend
    1.5 * np.cos(y / 200) +           # N-S trend
    0.5 * (z / 100) +                 # depth trend
    np.random.randn(n_samples) * 0.5  # noise
)

# Create DataFrame
data = pd.DataFrame({
    'East': x,
    'North': y,
    'Elevation': z,
    'Au_ppm': grade
})

print(f"Data shape: {data.shape}")
print(data.head())
```

## Define Variogram

```python
# Define variogram model
# (In practice, you'd use stereogamma to fit this from data)
variogram = {
    'model': 'spherical',
    'range': 300,      # range in meters
    'sill': 2.0,       # total variance
    'nugget': 0.25     # nugget effect
}
```

## Define Estimation Grid

```python
# Create regular block model grid
grid = {
    'xmin': 0, 'xmax': 1000, 'nx': 50,   # 20m blocks
    'ymin': 0, 'ymax': 1000, 'ny': 50,
    'zmin': 0, 'zmax': 200, 'nz': 20     # 10m blocks
}

print(f"Grid will have {50*50*20:,} blocks")
```

## Perform Ordinary Kriging

```python
# Simple kriging call
result = ggr.krige(
    data,
    variogram=variogram,
    grid=grid,
    method='ordinary',
    coordinate_columns=['East', 'North', 'Elevation'],
    value_column='Au_ppm',
    search_radius=500,
    max_points=30,
    min_points=4
)

print(f"Estimated {len(result):,} blocks")
print(f"Grade range: {np.nanmin(result):.2f} - {np.nanmax(result):.2f} ppm")
```

## Get Full Diagnostics

```python
# Get detailed output
result = ggr.krige(
    data,
    variogram=variogram,
    grid=grid,
    method='ordinary',
    coordinate_columns=['East', 'North', 'Elevation'],
    value_column='Au_ppm',
    return_diagnostics=True
)

# Examine results
print("Available outputs:")
for key in result.keys():
    print(f"  - {key}: shape {result[key].shape}")

# Check kriging quality
valid = ~np.isnan(result['estimates'])
print(f"\nKriging statistics:")
print(f"  Blocks estimated: {np.sum(valid):,} / {len(valid):,}")
print(f"  Mean efficiency: {np.nanmean(result['kriging_efficiency']):.3f}")
print(f"  Mean variance: {np.nanmean(result['variance']):.3f}")
```

## Visualize Results (2D Slice)

```python
# Reshape to grid (if regular)
nx, ny, nz = 50, 50, 20
estimates_3d = result['estimates'].reshape(nx, ny, nz)
variance_3d = result['variance'].reshape(nx, ny, nz)

# Plot mid-elevation slice
mid_z = nz // 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Estimates
im1 = ax1.imshow(
    estimates_3d[:, :, mid_z].T,
    origin='lower',
    extent=[0, 1000, 0, 1000],
    cmap='viridis'
)
ax1.scatter(data['East'], data['North'], c='red', s=20, alpha=0.5, label='Drill holes')
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.set_title(f'Kriged Grade at {mid_z*10}m Elevation')
ax1.legend()
plt.colorbar(im1, ax=ax1, label='Au (ppm)')

# Variance
im2 = ax2.imshow(
    variance_3d[:, :, mid_z].T,
    origin='lower',
    extent=[0, 1000, 0, 1000],
    cmap='plasma'
)
ax2.scatter(data['East'], data['North'], c='red', s=20, alpha=0.5, label='Drill holes')
ax2.set_xlabel('East (m)')
ax2.set_ylabel('North (m)')
ax2.set_title(f'Kriging Variance at {mid_z*10}m Elevation')
ax2.legend()
plt.colorbar(im2, ax=ax2, label='Variance')

plt.tight_layout()
plt.show()
```

## Using Class-Based API

```python
# Alternative: class-based API for more control
kriging = ggr.OrdinaryKrige(
    variogram=variogram,
    search_radius=500,
    max_points=30,
    min_points=4
)

# Fit the data
kriging.fit(
    data,
    coordinate_columns=['East', 'North', 'Elevation'],
    value_column='Au_ppm'
)

# Predict on grid
estimates = kriging.predict(grid)

print(f"Class API result: {len(estimates):,} estimates")
```

## Anisotropic Search

```python
# Define anisotropic search ellipsoid
search = ggr.AnisotropicSearch(
    radius=[500, 300, 100],  # major, semi-major, minor axes
    dip_direction=45,        # degrees
    dip=30,
    rake=0,
    octants=True,            # enforce octant search
    max_points=40,
    min_points=4
)

# Use in kriging
result = ggr.krige(
    data,
    variogram=variogram,
    grid=grid,
    search_radius=search,  # pass search object
    coordinate_columns=['East', 'North', 'Elevation'],
    value_column='Au_ppm'
)

print("Kriging with anisotropic search complete!")
```

## Next Steps

- Use `stereogamma` to model variograms from your data
- Try Simple Kriging if you know the mean
- Export results for further analysis
- Run in JupyterLite for browser-based geostatistics!
