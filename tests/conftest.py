"""
Shared pytest fixtures and configuration.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def simple_2d_data():
    """Generate simple 2D test data."""
    np.random.seed(42)
    n = 30
    x = np.random.rand(n) * 100
    y = np.random.rand(n) * 100
    # Create spatial structure
    values = np.sin(x/20) + np.cos(y/20) + np.random.randn(n) * 0.1
    return np.column_stack([x, y, values])


@pytest.fixture
def simple_3d_data():
    """Generate simple 3D test data."""
    np.random.seed(42)
    n = 50
    x = np.random.rand(n) * 100
    y = np.random.rand(n) * 100
    z = np.random.rand(n) * 50
    # Create spatial structure
    values = np.sin(x/20) + np.cos(y/20) + z/100 + np.random.randn(n) * 0.1
    return np.column_stack([x, y, z, values])


@pytest.fixture
def simple_2d_df(simple_2d_data):
    """Convert 2D data to DataFrame."""
    return pd.DataFrame(simple_2d_data, columns=['x', 'y', 'value'])


@pytest.fixture
def simple_3d_df(simple_3d_data):
    """Convert 3D data to DataFrame."""
    return pd.DataFrame(simple_3d_data, columns=['x', 'y', 'z', 'value'])


@pytest.fixture
def standard_variogram():
    """Standard variogram for testing."""
    return {
        'model': 'spherical',
        'range': 50,
        'sill': 1.0,
        'nugget': 0.1
    }


@pytest.fixture
def standard_2d_grid():
    """Standard 2D grid for testing."""
    return {
        'xmin': 0, 'xmax': 100, 'nx': 21,
        'ymin': 0, 'ymax': 100, 'ny': 21
    }


@pytest.fixture
def standard_3d_grid():
    """Standard 3D grid for testing."""
    return {
        'xmin': 0, 'xmax': 100, 'nx': 11,
        'ymin': 0, 'ymax': 100, 'ny': 11,
        'zmin': 0, 'zmax': 50, 'nz': 6
    }
