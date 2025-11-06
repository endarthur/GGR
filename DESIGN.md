# GGR Design Document

**Geostatistical Grid Realization - Technical Design**

Last Updated: November 2025

---

## 1. Project Vision

**Mission**: Bring professional geostatistics to the browser through a modern Python library that works seamlessly in JupyterLite.

**Name**: GGR - Officially "Geostatistical Grid Realization", secretly "Gamma Gamma Revolution" 💚

**Core Principle**: Zero-installation geostatistics accessible anywhere with a web browser.

---

## 2. Architecture Overview

### 2.1 Technology Stack

**Language**: Pure Python (3.9+)

**Core Dependencies** (JupyterLite-compatible):
- `numpy` >= 1.20 - Array operations
- `scipy` >= 1.7 - Spatial indexing (cKDTree), linear algebra
- `pandas` >= 1.3 - Data handling

**Optional Dependencies**:
- `matplotlib` >= 3.3 - Visualization

**NOT Compatible** (explicitly avoided):
- Numba (JIT compilation doesn't work in Pyodide)
- Cython (C extensions incompatible)
- Polars (Rust backend not fully Pyodide-compatible)
- Pyproj (heavy C dependencies)

### 2.2 Module Structure

```
ggr/
├── variogram.py    # Variogram models and covariance
├── search.py       # Anisotropic search strategies
├── grid.py         # Grid definitions
├── kriging.py      # Kriging algorithms (SK, OK)
└── (future)
    ├── simulation.py  # SGS, etc.
    └── validation.py  # Cross-validation tools
```

---

## 3. Design Decisions

### 3.1 Why Python Instead of WASM?

**Decision**: Implement from scratch in Python, not compile GSLIB to WASM.

**Rationale**:
- Target use cases won't run massive grids (millions of blocks acceptable)
- JavaScript/Python performance ~2-10x slower than WASM, but still fast enough
- Clean integration with existing Python geological libraries
- Easier to debug, modify, and extend
- RNG reproducibility maintained (ACORN implementation in Python)
- No WASM/WASI complexity

**Trade-off**: Sacrifice some performance for development speed and maintainability.

### 3.2 API Design Philosophy

**Dual API Strategy**: Both functional and class-based

**Functional API** (Quick & Easy):
```python
result = ggr.krige(data, variogram, grid, method='ordinary')
```

**Class-Based API** (More Control):
```python
krige = ggr.OrdinaryKrige(variogram=variogram)
krige.fit(data)
result = krige.predict(grid)
```

**Type Flexibility**: Accept multiple input formats
- Variogram: dict or Variogram object
- Grid: dict, Grid object, or numpy array
- Data: pandas DataFrame or numpy array

**Input/Output Matching**:
- DataFrame in → DataFrame out (future enhancement)
- NumPy array in → NumPy array out

### 3.3 Coordinate System Handling

**Decision**: No CRS/projection support in v1

**Default Convention**:
- First 3 columns (or 2 for 2D) are coordinates
- Override with `coordinate_columns` parameter
- Next column is values
- Override with `value_column` parameter

**Auto-detection**:
- 2D vs 3D from `len(coordinate_columns)` or array shape

**Rationale**: 
- Geostatistics works in local projected coordinates
- Pyproj unavailable in JupyterLite
- Users handle projections before/after using GGR
- Keeps implementation simple

### 3.4 Spatial Indexing

**Decision**: Use `scipy.spatial.cKDTree`

**Rationale**:
- Available in Pyodide ✅
- Fast, battle-tested
- Native 2D/3D support
- Radius searches built-in

**Rejected Alternatives**:
- Custom octree: More work for minimal gain
- No indexing: Too slow for practical use

### 3.5 Search Strategy

**Features**:
- Isotropic and anisotropic search
- Octant search enforcement (optional)
- Multiple angle conventions

**Angle Conventions Supported**:
1. **Dip direction, dip, rake** (default) - Structural geology standard
2. **Azimuth, dip, plunge** (GSLIB convention)
3. **Strike, dip, rake** (Alternative structural geology)

**Search Parameters**:
```python
search = AnisotropicSearch(
    radius=[500, 300, 100],  # major, semi, minor
    dip_direction=45,        # degrees
    dip=30,
    rake=0,
    octants=True,
    max_points=50,
    min_points=4
)
```

### 3.6 Grid Specification

**Two modes supported**:

1. **Regular Grid** (most common):
```python
grid = {
    'xmin': 0, 'xmax': 1000, 'nx': 100,
    'ymin': 0, 'ymax': 1000, 'ny': 100,
    'zmin': 0, 'zmax': 100, 'nz': 10
}
```

2. **Arbitrary Points**:
```python
grid = np.array([[x1,y1,z1], [x2,y2,z2], ...])
```

### 3.7 Kriging Outputs

**Standard Output**: Just estimates (numpy array)

**Optional Outputs** (via `return_variance=True` or `return_diagnostics=True`):
```python
result = {
    'estimates': array,
    'variance': array,
    'n_samples': array,             # samples used per point
    'mean_distance': array,         # avg distance to samples
    'slope_of_regression': array,   # should be ~1 for OK
    'kriging_efficiency': array,    # 1 - variance/sill
    'lagrange_multiplier': array,   # for OK
}
```

### 3.8 Error Handling Strategy

**Fail Fast**:
- Invalid variogram parameters
- No data provided
- Dimension mismatches

**Warn**:
- Few samples (< min_points)
- Large search radius
- Suspect angle values

**Silent/Flag**:
- No data in search radius → NaN in output
- Singular matrix → NaN in output (with optional variance = -1)

---

## 4. Implementation Priorities

### Phase 1: Core Kriging (v0.1)
1. ✅ Project structure
2. ⏳ Variogram covariance (spherical, exponential, gaussian)
3. ⏳ Simple Kriging (SK)
4. ⏳ Ordinary Kriging (OK)
5. ⏳ Isotropic search
6. ⏳ GSLIB validation

### Phase 2: Anisotropy & Diagnostics (v0.2)
1. ⏳ Anisotropic search with rotation
2. ⏳ Octant search
3. ⏳ Full diagnostic outputs
4. ⏳ Angle convention converters

### Phase 3: Simulation (v0.3)
1. ⏳ ACORN RNG implementation
2. ⏳ Sequential Gaussian Simulation (SGS)
3. ⏳ Multiple realizations

### Phase 4: Advanced (v0.4+)
1. ⏳ Universal Kriging
2. ⏳ Co-kriging
3. ⏳ Indicator Kriging
4. ⏳ Block kriging (discretization)

---

## 5. Testing Strategy

### 5.1 Test Organization

Three test categories:

1. **API Contract Tests** (`test_api.py`)
   - Interface behavior
   - Input format handling
   - Output format verification
   - Parameter validation

2. **GSLIB Validation Tests** (`test_gslib_validation.py`)
   - Numerical correctness
   - Compare against GSLIB outputs
   - Standard test datasets
   - Target tolerance: <1% relative error

3. **Edge Case Tests** (`test_edge_cases.py`)
   - Error handling
   - Boundary conditions
   - Singular matrices
   - No data scenarios

### 5.2 TDD Workflow

1. Write test first (red)
2. Implement minimum code (green)
3. Refactor (keep green)
4. Repeat

**Test-First Mandate**: No implementation without tests.

### 5.3 GSLIB Validation

**Reference Datasets**:
- cluster.dat (standard GSLIB dataset)
- ydata.dat (well-spaced data)
- Custom synthetic datasets

**Validation Process**:
1. Run GSLIB program (kt3d, kb2d, etc.)
2. Save output
3. Run GGR with same parameters
4. Compare: `assert_allclose(ggr_result, gslib_result, rtol=0.01)`

**Acceptance**: Results within 1% relative tolerance

---

## 6. Algorithmic Decisions

### 6.1 Variogram Models

**Models to Implement**:
- Spherical (most common)
- Exponential
- Gaussian
- Linear (simple)
- Power (for trends)

**Covariance Calculation**:
```python
def covariance(h: float) -> float:
    """Calculate covariance at distance h"""
    if h == 0:
        return sill  # No nugget at zero distance
    
    # Model-specific calculation
    gamma = variogram_value(h)
    return sill - gamma
```

**Anisotropy**: Apply rotation matrix to distance vectors before calculating covariance.

### 6.2 Kriging System

**Ordinary Kriging**:
```
[C11  C12  ... C1n  1  ] [w1 ]   [C10]
[C21  C22  ... C2n  1  ] [w2 ]   [C20]
[...  ...  ... ...  .. ] [.. ] = [.. ]
[Cn1  Cn2  ... Cnn  1  ] [wn ]   [Cn0]
[1    1    ... 1    0  ] [μ  ]   [1  ]
```

Where:
- Cij = covariance between data points i and j
- Ci0 = covariance between data point i and estimation point
- wi = kriging weights
- μ = Lagrange multiplier (unbiasedness constraint)

**Simple Kriging** (no Lagrange multiplier):
```
[C11  C12  ... C1n ] [w1 ]   [C10]
[C21  C22  ... C2n ] [w2 ]   [C20]
[...  ...  ... ... ] [.. ] = [.. ]
[Cn1  Cn2  ... Cnn ] [wn ]   [Cn0]
```

**Solver**: Use `scipy.linalg.solve()` for efficiency

### 6.3 Rotation Matrices

**3D Rotation** (ZXZ Euler convention):
```python
def build_rotation_matrix(dip_direction, dip, rake):
    """
    Build 3D rotation matrix.
    
    Order: Z-rotation (dip_direction) → X-rotation (dip) → Z-rotation (rake)
    """
    # Convert to radians
    dd = radians(dip_direction)
    d = radians(dip)
    r = radians(rake)
    
    # Build individual rotation matrices
    Rz1 = [[cos(dd), -sin(dd), 0],
           [sin(dd),  cos(dd), 0],
           [0,        0,       1]]
    
    Rx = [[1, 0,       0      ],
          [0, cos(d), -sin(d)],
          [0, sin(d),  cos(d)]]
    
    Rz2 = [[cos(r), -sin(r), 0],
           [sin(r),  cos(r), 0],
           [0,       0,      1]]
    
    # Combine: R = Rz1 @ Rx @ Rz2
    return matrix_multiply(Rz1, Rx, Rz2)
```

### 6.4 Neighbor Selection

**Algorithm**:
1. Query KDTree for points within search radius
2. If anisotropic, transform coordinates and filter by ellipsoid
3. If octant search, enforce minimum per octant
4. Limit to max_points
5. Check min_points requirement

**Octant Search**:
- Divide 3D space into 8 octants
- Try to get at least 1-2 points from each octant
- Improves directional coverage

---

## 7. Performance Considerations

### 7.1 Expected Performance

**Target**: 1-10ms per kriging point (Python)

**Typical Block Model**:
- 100 x 100 x 50 = 500,000 blocks
- 30 data points per block (average)
- Small kriging systems (30x30 matrices)
- Estimated time: 5-50 minutes single-threaded

**Optimization Strategies**:
1. Spatial indexing (KDTree) - critical
2. Vectorized operations where possible
3. Efficient matrix solving (scipy)
4. (Future) Web Workers for parallelization

### 7.2 Scalability Limits

**Practical Limits** (in browser):
- Block models: ~1-5 million blocks
- Data points: ~10,000-100,000
- Search neighbors: 50-100 max

**When to use native tools**: >10M blocks or >100K data points

---

## 8. Future Enhancements

### Post v1.0

**Performance**:
- Web Workers for parallelization
- Progressive rendering
- Chunked processing for large grids

**Features**:
- Cross-validation tools
- Variogram fitting (integrate with stereogamma)
- Conditional simulation
- Indicator methods
- Trend modeling

**Integration**:
- Export to VTK for visualization
- Integration with geological modeling libraries
- JupyterLite-specific optimizations

---

## 9. Known Limitations

**By Design**:
- No CRS/projection support (use external tools)
- No parallel processing in v1 (single-threaded)
- Pure Python (slower than compiled code)
- Limited to Pyodide-compatible dependencies

**Acceptable Trade-offs**:
- 2-10x slower than GSLIB (still fast enough for target use cases)
- No GPU acceleration (browser limitations)
- Memory limits (browser sandbox constraints)

---

## 10. Success Criteria

**Technical**:
- ✅ Works in JupyterLite
- ✅ Matches GSLIB within 1%
- ✅ Handles 1M+ block models
- ✅ Clean, tested, documented code

**User Experience**:
- ✅ Zero installation required
- ✅ Intuitive API
- ✅ Works on tablets/Chromebooks
- ✅ Clear error messages

**Community**:
- ✅ Open source (MIT)
- ✅ Well-documented
- ✅ Easy to contribute
- ✅ Educational value

---

## 11. Related Projects

**Stereogamma**: Companion variography library (separate project)
- Experimental variogram calculation
- Variogram model fitting
- GUI components for variography
- Will provide Variogram objects to GGR

**Integration**: GGR accepts stereogamma Variogram objects natively

---

## 12. References

**Geostatistics Theory**:
- Deutsch & Journel (1998): "GSLIB: Geostatistical Software Library and User's Guide"
- Isaaks & Srivastava (1989): "Applied Geostatistics"

**GSLIB**: http://www.statios.com/Quick/gslib.html

**Technical**:
- Pyodide: https://pyodide.org/
- JupyterLite: https://jupyterlite.readthedocs.io/
- SciPy spatial: https://docs.scipy.org/doc/scipy/reference/spatial.html

---

**Document History**:
- 2025-11: Initial design document created
- Future updates will be tracked here

**Gamma Gamma Revolution** 💚
