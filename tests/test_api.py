"""
API contract tests.

Tests that the API works as documented and handles various input formats.
"""

import pytest
import numpy as np
import pandas as pd
import ggr


class TestVariogramAPI:
    """Test Variogram class API."""
    
    def test_create_from_params(self):
        """Test creating variogram from parameters."""
        vario = ggr.Variogram('spherical', range_=500, sill=1.0, nugget=0.1)
        assert vario.model == 'spherical'
        assert vario.range[0] == 500
        assert vario.sill == 1.0
        assert vario.nugget == 0.1
    
    def test_create_from_dict(self):
        """Test creating variogram from dict."""
        vario = ggr.Variogram.from_dict({
            'model': 'spherical',
            'range': 500,
            'sill': 1.0,
            'nugget': 0.1
        })
        assert vario.model == 'spherical'
        assert vario.sill == 1.0
    
    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            ggr.Variogram('spherical', range_=500, sill=-1.0)  # negative sill
        
        with pytest.raises(ValueError):
            ggr.Variogram('spherical', range_=-500, sill=1.0)  # negative range


class TestGridAPI:
    """Test Grid class API."""
    
    def test_create_regular_2d(self):
        """Test creating 2D regular grid."""
        grid = ggr.Grid(xmin=0, xmax=100, nx=11,
                        ymin=0, ymax=100, ny=11)
        assert grid.ndim == 2
        assert grid.is_regular
        assert grid.n_points == 11 * 11
    
    def test_create_regular_3d(self):
        """Test creating 3D regular grid."""
        grid = ggr.Grid(xmin=0, xmax=100, nx=11,
                        ymin=0, ymax=100, ny=11,
                        zmin=0, zmax=50, nz=6)
        assert grid.ndim == 3
        assert grid.is_regular
        assert grid.n_points == 11 * 11 * 6
    
    def test_create_from_points(self):
        """Test creating grid from arbitrary points."""
        points = np.random.rand(100, 3)
        grid = ggr.Grid(points=points)
        assert grid.ndim == 3
        assert not grid.is_regular
        assert grid.n_points == 100
    
    def test_create_from_dict(self):
        """Test creating grid from dict."""
        grid = ggr.Grid.from_dict({
            'xmin': 0, 'xmax': 100, 'nx': 11,
            'ymin': 0, 'ymax': 100, 'ny': 11
        })
        assert grid.ndim == 2
        assert grid.n_points == 121


class TestSearchAPI:
    """Test AnisotropicSearch API."""
    
    def test_create_isotropic(self):
        """Test isotropic search."""
        search = ggr.AnisotropicSearch(radius=500)
        assert search.is_isotropic()
    
    def test_create_anisotropic(self):
        """Test anisotropic search."""
        search = ggr.AnisotropicSearch(
            radius=[500, 300, 100],
            dip_direction=45, dip=30, rake=0
        )
        assert not search.is_isotropic()
        assert search.octants == False
    
    def test_angle_conventions(self):
        """Test different angle conventions."""
        # Default convention
        search1 = ggr.AnisotropicSearch(
            radius=500,
            dip_direction=45, dip=30, rake=0
        )
        
        # GSLIB convention
        search2 = ggr.AnisotropicSearch(
            radius=500,
            dip_direction=45, dip=30, rake=0,
            convention='gslib'
        )
        
        # Should create valid searches
        assert search1.dip_direction == 45
        assert search2._convention == 'gslib'


class TestKrigingAPI:
    """Test kriging API."""
    
    @pytest.fixture
    def sample_data_array(self):
        """Create sample data as NumPy array."""
        np.random.seed(42)
        n = 50
        x = np.random.rand(n) * 100
        y = np.random.rand(n) * 100
        z = np.random.rand(n) * 50
        values = np.random.rand(n) * 10
        return np.column_stack([x, y, z, values])
    
    @pytest.fixture
    def sample_data_df(self, sample_data_array):
        """Create sample data as DataFrame."""
        return pd.DataFrame(
            sample_data_array,
            columns=['x', 'y', 'z', 'grade']
        )
    
    @pytest.fixture
    def sample_variogram(self):
        """Create sample variogram."""
        return {
            'model': 'spherical',
            'range': 50,
            'sill': 1.0,
            'nugget': 0.1
        }
    
    @pytest.fixture
    def sample_grid(self):
        """Create sample grid."""
        return {
            'xmin': 0, 'xmax': 100, 'nx': 11,
            'ymin': 0, 'ymax': 100, 'ny': 11,
            'zmin': 0, 'zmax': 50, 'nz': 6
        }
    
    def test_functional_api_array(self, sample_data_array, sample_variogram, sample_grid):
        """Test functional API with NumPy array."""
        result = ggr.krige(
            sample_data_array,
            sample_variogram,
            sample_grid,
            method='ordinary'
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 11 * 11 * 6
    
    def test_functional_api_dataframe(self, sample_data_df, sample_variogram, sample_grid):
        """Test functional API with DataFrame."""
        result = ggr.krige(
            sample_data_df,
            sample_variogram,
            sample_grid,
            method='ordinary'
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 11 * 11 * 6
    
    def test_class_api(self, sample_data_array, sample_variogram, sample_grid):
        """Test class-based API."""
        kriging = ggr.OrdinaryKrige(sample_variogram)
        kriging.fit(sample_data_array)
        result = kriging.predict(sample_grid)
        assert isinstance(result, np.ndarray)
    
    def test_return_variance(self, sample_data_array, sample_variogram, sample_grid):
        """Test returning variance."""
        result = ggr.krige(
            sample_data_array,
            sample_variogram,
            sample_grid,
            return_variance=True
        )
        assert isinstance(result, dict)
        assert 'estimates' in result
        assert 'variance' in result
    
    def test_return_diagnostics(self, sample_data_array, sample_variogram, sample_grid):
        """Test returning full diagnostics."""
        result = ggr.krige(
            sample_data_array,
            sample_variogram,
            sample_grid,
            return_diagnostics=True
        )
        assert isinstance(result, dict)
        assert 'estimates' in result
        assert 'variance' in result
        assert 'n_samples' in result
        assert 'kriging_efficiency' in result
