"""
Kriging algorithms.

Implements Simple Kriging (SK) and Ordinary Kriging (OK) with support
for anisotropic search and comprehensive diagnostics.
"""

from typing import Union, Dict, Optional, Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.spatial import cKDTree

from .variogram import Variogram, parse_variogram
from .search import AnisotropicSearch
from .grid import Grid, parse_grid


class BaseKrige:
    """
    Base class for kriging algorithms.
    
    Provides common functionality for all kriging variants.
    """
    
    def __init__(
        self,
        variogram: Union[Variogram, Dict],
        search_radius: Union[float, AnisotropicSearch] = 500.0,
        max_points: int = 50,
        min_points: int = 4,
    ):
        self.variogram = parse_variogram(variogram)
        
        # Parse search
        if isinstance(search_radius, (int, float)):
            self.search = AnisotropicSearch(
                radius=search_radius,
                max_points=max_points,
                min_points=min_points,
            )
        elif isinstance(search_radius, AnisotropicSearch):
            self.search = search_radius
        else:
            raise TypeError("search_radius must be float or AnisotropicSearch")
        
        # Store data
        self.data_coords = None
        self.data_values = None
        self.kdtree = None
    
    def fit(self, data: Union[NDArray, pd.DataFrame], coordinate_columns=None, value_column=None):
        """
        Fit kriging with data.
        
        Parameters
        ----------
        data : ndarray or DataFrame
            Input data with coordinates and values
        coordinate_columns : list of str, optional
            Column names for coordinates (for DataFrame input)
        value_column : str, optional
            Column name for values (for DataFrame input)
        """
        # Parse input data
        self.data_coords, self.data_values = self._parse_data(
            data, coordinate_columns, value_column
        )
        
        # Build spatial index
        self.kdtree = cKDTree(self.data_coords)
    
    def _parse_data(
        self,
        data: Union[NDArray, pd.DataFrame],
        coordinate_columns=None,
        value_column=None
    ):
        """Parse input data into coordinates and values."""
        if isinstance(data, pd.DataFrame):
            # DataFrame input
            if coordinate_columns is None:
                # Use first 2 or 3 columns
                n_coords = 3 if len(data.columns) >= 4 else 2
                coordinate_columns = data.columns[:n_coords].tolist()
            
            if value_column is None:
                # Use column after coordinates
                value_column = data.columns[len(coordinate_columns)]
            
            coords = data[coordinate_columns].values
            values = data[value_column].values
        
        elif isinstance(data, np.ndarray):
            # NumPy array input
            if data.ndim != 2:
                raise ValueError("Data array must be 2D")
            
            # Assume first n columns are coords, last is value
            n_coords = data.shape[1] - 1
            coords = data[:, :n_coords]
            values = data[:, -1]
        
        else:
            raise TypeError("Data must be DataFrame or ndarray")
        
        return coords, values
    
    def predict(
        self,
        grid: Union[Grid, Dict, NDArray],
        return_variance: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[NDArray, Dict]:
        """
        Perform kriging prediction.
        
        Parameters
        ----------
        grid : Grid, dict, or ndarray
            Points where to estimate
        return_variance : bool
            Return kriging variance
        return_diagnostics : bool
            Return full diagnostics
        
        Returns
        -------
        ndarray or dict
            Estimates (if no extra outputs requested), or dict with results
        """
        if self.data_coords is None:
            raise ValueError("Must call fit() before predict()")
        
        # Parse grid
        grid_obj = parse_grid(grid)
        points = grid_obj.points
        
        # Initialize outputs
        n_points = len(points)
        estimates = np.full(n_points, np.nan)
        
        if return_variance or return_diagnostics:
            variances = np.full(n_points, np.nan)
        if return_diagnostics:
            n_samples = np.zeros(n_points, dtype=int)
            mean_distances = np.full(n_points, np.nan)
            slopes = np.full(n_points, np.nan)
            efficiencies = np.full(n_points, np.nan)
        
        # Krige each point
        for i, point in enumerate(points):
            result = self._krige_point(point, return_diagnostics)
            
            estimates[i] = result['estimate']
            if return_variance or return_diagnostics:
                variances[i] = result.get('variance', np.nan)
            if return_diagnostics:
                n_samples[i] = result.get('n_samples', 0)
                mean_distances[i] = result.get('mean_distance', np.nan)
                slopes[i] = result.get('slope', np.nan)
                efficiencies[i] = result.get('efficiency', np.nan)
        
        # Return results
        if not (return_variance or return_diagnostics):
            return estimates
        
        result_dict = {'estimates': estimates}
        
        if return_variance:
            result_dict['variance'] = variances
        
        if return_diagnostics:
            result_dict.update({
                'variance': variances,
                'n_samples': n_samples,
                'mean_distance': mean_distances,
                'slope_of_regression': slopes,
                'kriging_efficiency': efficiencies,
            })
        
        return result_dict
    
    def _krige_point(self, point: NDArray, return_diagnostics: bool = False) -> Dict:
        """
        Krige a single point.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement _krige_point")


class SimpleKrige(BaseKrige):
    """
    Simple Kriging (SK).
    
    Assumes known stationary mean.
    
    Parameters
    ----------
    variogram : Variogram or dict
        Variogram model
    mean : float
        Known mean value
    search_radius : float or AnisotropicSearch
        Search strategy
    max_points : int
        Maximum data points to use
    min_points : int
        Minimum data points required
    """
    
    def __init__(
        self,
        variogram: Union[Variogram, Dict],
        mean: float,
        search_radius: Union[float, AnisotropicSearch] = 500.0,
        max_points: int = 50,
        min_points: int = 4,
    ):
        super().__init__(variogram, search_radius, max_points, min_points)
        self.mean = mean
    
    def _krige_point(self, point: NDArray, return_diagnostics: bool = False) -> Dict:
        """Perform Simple Kriging at a single point."""
        # TODO: Implement Simple Kriging algorithm
        # 1. Find neighbors
        # 2. Build covariance matrix
        # 3. Solve kriging system
        # 4. Calculate estimate and variance
        
        result = {
            'estimate': np.nan,
            'variance': np.nan,
        }
        
        if return_diagnostics:
            result.update({
                'n_samples': 0,
                'mean_distance': np.nan,
                'slope': np.nan,
                'efficiency': np.nan,
            })
        
        return result


class OrdinaryKrige(BaseKrige):
    """
    Ordinary Kriging (OK).
    
    Estimates local mean from data (most common kriging variant).
    
    Parameters
    ----------
    variogram : Variogram or dict
        Variogram model
    search_radius : float or AnisotropicSearch
        Search strategy
    max_points : int
        Maximum data points to use
    min_points : int
        Minimum data points required
    """
    
    def __init__(
        self,
        variogram: Union[Variogram, Dict],
        search_radius: Union[float, AnisotropicSearch] = 500.0,
        max_points: int = 50,
        min_points: int = 4,
    ):
        super().__init__(variogram, search_radius, max_points, min_points)
    
    def _krige_point(self, point: NDArray, return_diagnostics: bool = False) -> Dict:
        """Perform Ordinary Kriging at a single point."""
        # TODO: Implement Ordinary Kriging algorithm
        # 1. Find neighbors
        # 2. Build covariance matrix (with Lagrange multiplier)
        # 3. Solve kriging system
        # 4. Calculate estimate and variance
        
        result = {
            'estimate': np.nan,
            'variance': np.nan,
        }
        
        if return_diagnostics:
            result.update({
                'n_samples': 0,
                'mean_distance': np.nan,
                'slope': np.nan,
                'efficiency': np.nan,
            })
        
        return result


# Functional API

def krige(
    data: Union[NDArray, pd.DataFrame],
    variogram: Union[Variogram, Dict],
    grid: Union[Grid, Dict, NDArray],
    method: Literal['simple', 'ordinary'] = 'ordinary',
    mean: Optional[float] = None,
    search_radius: Union[float, AnisotropicSearch] = 500.0,
    max_points: int = 50,
    min_points: int = 4,
    coordinate_columns=None,
    value_column=None,
    return_variance: bool = False,
    return_diagnostics: bool = False,
) -> Union[NDArray, Dict]:
    """
    Perform kriging estimation (functional API).
    
    Parameters
    ----------
    data : ndarray or DataFrame
        Input data with coordinates and values
    variogram : Variogram or dict
        Variogram model
    grid : Grid, dict, or ndarray
        Estimation grid
    method : {'simple', 'ordinary'}
        Kriging method
    mean : float, optional
        Known mean (required for Simple Kriging)
    search_radius : float or AnisotropicSearch
        Search strategy
    max_points : int
        Maximum data points to use
    min_points : int
        Minimum data points required
    coordinate_columns : list of str, optional
        Column names for coordinates (DataFrame only)
    value_column : str, optional
        Column name for values (DataFrame only)
    return_variance : bool
        Return kriging variance
    return_diagnostics : bool
        Return full diagnostics
    
    Returns
    -------
    ndarray or dict
        Estimates, or dict with estimates and diagnostics
    
    Examples
    --------
    >>> result = krige(data, variogram, grid, method='ordinary')
    >>> result = krige(data, variogram, grid, method='simple', mean=1.5)
    """
    if method == 'simple':
        if mean is None:
            raise ValueError("mean must be provided for Simple Kriging")
        kriging = SimpleKrige(variogram, mean, search_radius, max_points, min_points)
    elif method == 'ordinary':
        kriging = OrdinaryKrige(variogram, search_radius, max_points, min_points)
    else:
        raise ValueError(f"Unknown kriging method: {method}")
    
    kriging.fit(data, coordinate_columns, value_column)
    return kriging.predict(grid, return_variance, return_diagnostics)


def simple_krige(*args, **kwargs):
    """Simple Kriging (convenience function)."""
    kwargs['method'] = 'simple'
    return krige(*args, **kwargs)


def ordinary_krige(*args, **kwargs):
    """Ordinary Kriging (convenience function)."""
    kwargs['method'] = 'ordinary'
    return krige(*args, **kwargs)
