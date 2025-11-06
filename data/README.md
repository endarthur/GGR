# Data Directory

This directory contains test datasets and GSLIB reference outputs for validation.

## Structure

```
data/
├── samples/          # Sample datasets for examples
│   └── synthetic_drillholes.csv
├── gslib/           # GSLIB reference data and outputs
│   ├── cluster.dat  # Standard GSLIB test dataset
│   ├── kt3d.par     # GSLIB parameter files
│   └── kt3d.out     # GSLIB output for validation
└── validation/      # Validation test results
```

## Adding Test Data

When adding validation data:

1. Include original GSLIB data files
2. Include GSLIB parameter files (.par)
3. Include GSLIB output files for comparison
4. Document any preprocessing steps
5. Add data description to this README

## GSLIB Test Datasets

Standard GSLIB datasets for testing:

- **cluster.dat** - Clustered point data (140 samples, 2D)
- **ydata.dat** - Well-spaced data for kriging
- **true.dat** - Exhaustive dataset for validation

Download from: [GSLIB website](http://www.statios.com/Quick/gslib.html)

## Example Datasets

Coming soon:
- Synthetic drill hole data (Au grades)
- Regular grid for testing
- Anisotropic test case

## Usage

```python
import pandas as pd

# Load sample data
data = pd.read_csv('data/samples/synthetic_drillholes.csv')
```

## Notes

- Keep file sizes reasonable (<10 MB)
- Use CSV format for portability
- Document coordinate systems
- Include data dictionary
