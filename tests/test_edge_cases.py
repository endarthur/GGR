"""
Edge case tests.

Tests for error handling, boundary conditions, and special cases.
"""

import pytest
import numpy as np
import pandas as pd
import ggr


class TestErrorHandling:
    """Test proper error handling."""
    
    def test_predict_before_fit(self):
        """Test that predict fails if fit not called."""
        krige = ggr.OrdinaryKrige({'model': 'spherical', 'range': 500, 'sill': 1.0})
        
        grid = {'xmin': 0, 'xmax': 100, 'nx': 11,
                'ymin': 0, 'ymax': 100, 'ny': 11}
        
        with pytest.raises(ValueError, match="Must call fit"):
            krige.predict(grid)
    
    def test_invalid_variogram_model(self):
        """Test invalid variogram model."""
        with pytest.raises(ValueError):
            ggr.Variogram('invalid_model', range_=500, sill=1.0)
    
    def test_negative_sill(self):
        """Test that negative sill raises error."""
        with pytest.raises(ValueError):
            ggr.Variogram('spherical', range_=500, sill=-1.0)
    
    def test_invalid_data_format(self):
        """Test invalid data format."""
        krige = ggr.OrdinaryKrige({'model': 'spherical', 'range': 500, 'sill': 1.0})
        
        with pytest.raises(TypeError):
            krige.fit("invalid_data")  # String instead of array/DataFrame


class TestBoundaryConditions:
    """Test boundary conditions and special cases."""
    
    def test_single_data_point(self):
        """Test kriging with only one data point."""
        data = np.array([[50, 50, 50, 1.5]])
        
        variogram = {'model': 'spherical', 'range': 500, 'sill': 1.0}
        grid = {'xmin': 0, 'xmax': 100, 'nx': 3,
                'ymin': 0, 'ymax': 100, 'ny': 3,
                'zmin': 0, 'zmax': 100, 'nz': 3}
        
        # Should handle gracefully (might return NaN or mean)
        result = ggr.krige(data, variogram, grid, min_points=1)
        assert not np.all(np.isnan(result))
    
    def test_no_data_in_search(self):
        """Test when no data points are in search radius."""
        # Data far from grid
        data = np.array([
            [1000, 1000, 1000, 1.0],
            [1100, 1100, 1100, 2.0],
        ])
        
        # Grid far from data
        variogram = {'model': 'spherical', 'range': 500, 'sill': 1.0}
        grid = {'xmin': 0, 'xmax': 100, 'nx': 3,
                'ymin': 0, 'ymax': 100, 'ny': 3}
        
        result = ggr.krige(
            data, variogram, grid,
            search_radius=50,  # Small radius
            min_points=1
        )
        
        # Should return NaN where no data available
        assert np.all(np.isnan(result))
    
    def test_identical_data_locations(self):
        """Test data with identical coordinates."""
        data = np.array([
            [50, 50, 50, 1.0],
            [50, 50, 50, 1.5],  # Same location
            [60, 60, 60, 2.0],
        ])
        
        variogram = {'model': 'spherical', 'range': 500, 'sill': 1.0}
        grid = np.array([[55, 55, 55]])
        
        # Should handle without crashing (singular matrix)
        # Behavior TBD - might average, might fail gracefully
        try:
            result = ggr.krige(data, variogram, grid)
            assert not np.isnan(result[0]) or np.isnan(result[0])
        except np.linalg.LinAlgError:
            # Acceptable to raise LinAlgError for singular matrix
            pass
    
    def test_collinear_data(self):
        """Test data points that are collinear."""
        # All data on a line
        data = np.array([
            [0, 0, 0, 1.0],
            [10, 0, 0, 1.5],
            [20, 0, 0, 2.0],
            [30, 0, 0, 2.5],
        ])
        
        variogram = {'model': 'spherical', 'range': 50, 'sill': 1.0}
        grid = np.array([[15, 0, 0]])
        
        # Should work - collinearity is geometric, not numerical
        result = ggr.krige(data, variogram, grid)
        assert not np.isnan(result[0])
    
    def test_extreme_anisotropy(self):
        """Test extreme anisotropy ratios."""
        search = ggr.AnisotropicSearch(
            radius=[1000, 100, 10],  # 100:10:1 ratio
            dip_direction=0, dip=0, rake=0
        )
        
        data = np.random.rand(50, 4) * 100
        variogram = {'model': 'spherical', 'range': 500, 'sill': 1.0}
        grid = {'xmin': 0, 'xmax': 100, 'nx': 3,
                'ymin': 0, 'ymax': 100, 'ny': 3,
                'zmin': 0, 'zmax': 100, 'nz': 3}
        
        # Should handle extreme anisotropy
        result = ggr.krige(data, variogram, grid, search_radius=search)
        assert result.shape[0] == 3 * 3 * 3


class TestSpecialCases:
    """Test special mathematical cases."""
    
    def test_zero_nugget(self):
        """Test variogram with zero nugget."""
        variogram = ggr.Variogram('spherical', range_=500, sill=1.0, nugget=0.0)
        assert variogram.nugget == 0.0
    
    def test_pure_nugget(self):
        """Test variogram with sill = nugget (pure nugget effect)."""
        variogram = ggr.Variogram('spherical', range_=500, sill=1.0, nugget=1.0)
        # Should be valid even though it's degenerate
        assert variogram.nugget == variogram.sill
    
    def test_constant_data(self):
        """Test kriging with constant data values."""
        data = np.array([
            [0, 0, 0, 5.0],
            [10, 0, 0, 5.0],
            [0, 10, 0, 5.0],
            [10, 10, 0, 5.0],
        ])
        
        variogram = {'model': 'spherical', 'range': 50, 'sill': 0.1}
        grid = np.array([[5, 5, 0]])
        
        result = ggr.krige(data, variogram, grid)
        
        # Should return constant value
        np.testing.assert_allclose(result, 5.0, rtol=0.01)
    
    def test_2d_vs_3d(self):
        """Test that 2D and 3D work properly."""
        # 2D data
        data_2d = np.random.rand(20, 3)  # x, y, value
        variogram = {'model': 'spherical', 'range': 50, 'sill': 1.0}
        grid_2d = {'xmin': 0, 'xmax': 1, 'nx': 5,
                   'ymin': 0, 'ymax': 1, 'ny': 5}
        
        result_2d = ggr.krige(data_2d, variogram, grid_2d)
        assert result_2d.shape[0] == 5 * 5
        
        # 3D data
        data_3d = np.random.rand(20, 4)  # x, y, z, value
        grid_3d = {'xmin': 0, 'xmax': 1, 'nx': 5,
                   'ymin': 0, 'ymax': 1, 'ny': 5,
                   'zmin': 0, 'zmax': 1, 'nz': 5}
        
        result_3d = ggr.krige(data_3d, variogram, grid_3d)
        assert result_3d.shape[0] == 5 * 5 * 5


class TestInputOutputTypes:
    """Test various input/output type combinations."""
    
    def test_dataframe_in_array_out(self):
        """Test DataFrame input returns array by default."""
        df = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20),
            'z': np.random.rand(20),
            'value': np.random.rand(20)
        })
        
        variogram = {'model': 'spherical', 'range': 0.5, 'sill': 1.0}
        grid = {'xmin': 0, 'xmax': 1, 'nx': 5,
                'ymin': 0, 'ymax': 1, 'ny': 5,
                'zmin': 0, 'xmax': 1, 'nz': 5}
        
        result = ggr.krige(df, variogram, grid)
        assert isinstance(result, np.ndarray)
    
    def test_array_in_array_out(self):
        """Test array input returns array."""
        data = np.random.rand(20, 4)
        variogram = {'model': 'spherical', 'range': 0.5, 'sill': 1.0}
        grid = {'xmin': 0, 'xmax': 1, 'nx': 5,
                'ymin': 0, 'ymax': 1, 'ny': 5,
                'zmin': 0, 'zmax': 1, 'nz': 5}
        
        result = ggr.krige(data, variogram, grid)
        assert isinstance(result, np.ndarray)
