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

        # Handle dimensionality mismatch between data and grid
        data_ndim = self.data_coords.shape[1]
        grid_ndim = points.shape[1]

        if grid_ndim < data_ndim:
            # Grid is lower dimension than data (e.g., 2D grid, 3D data)
            # Pad grid points with zeros
            padding = np.zeros((len(points), data_ndim - grid_ndim))
            points = np.column_stack([points, padding])
        elif grid_ndim > data_ndim:
            # Grid is higher dimension than data
            raise ValueError(
                f"Grid dimensionality ({grid_ndim}) exceeds data dimensionality ({data_ndim}). "
                f"Cannot estimate {grid_ndim}D points from {data_ndim}D data."
            )

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
        """
        Perform Simple Kriging at a single point.

        Simple Kriging system:
        C * w = C0

        Where:
        - C is the covariance matrix between data points (n x n)
        - w is the weights vector (n x 1)
        - C0 is the covariance vector between data and estimation point (n x 1)

        Estimate: z*(x0) = mean + sum(wi * (zi - mean))
        Variance: sigma^2 = C(0,0) - sum(wi * C0i)
        """
        # 1. Find neighbors using isotropic search
        # For now, we just use radius search from kdtree
        max_radius = self.search.radius[0] if len(self.search.radius) > 0 else self.search.radius

        # Query neighbors within search radius
        indices = self.kdtree.query_ball_point(point, r=max_radius)

        # Check minimum points requirement
        if len(indices) < self.search.min_points:
            result = {
                'estimate': np.nan,
                'variance': np.nan,
            }
            if return_diagnostics:
                result.update({
                    'n_samples': len(indices),
                    'mean_distance': np.nan,
                    'slope': np.nan,
                    'efficiency': np.nan,
                })
            return result

        # Limit to max_points (use closest if too many)
        if len(indices) > self.search.max_points:
            # Calculate distances and sort
            dists = np.linalg.norm(self.data_coords[indices] - point, axis=1)
            sorted_idx = np.argsort(dists)
            indices = [indices[i] for i in sorted_idx[:self.search.max_points]]

        # Get neighbor coordinates and values
        neighbor_coords = self.data_coords[indices]
        neighbor_values = self.data_values[indices]
        n_neighbors = len(indices)

        # 2. Build covariance matrix C (between data points)
        C = np.zeros((n_neighbors, n_neighbors))
        for i in range(n_neighbors):
            for j in range(i, n_neighbors):
                # Calculate distance
                h = np.linalg.norm(neighbor_coords[i] - neighbor_coords[j])
                # Calculate covariance
                cov = self.variogram.covariance(np.array([h]))[0]
                C[i, j] = cov
                C[j, i] = cov  # Symmetric

        # 3. Build covariance vector C0 (between data and estimation point)
        C0 = np.zeros(n_neighbors)
        for i in range(n_neighbors):
            h = np.linalg.norm(neighbor_coords[i] - point)
            C0[i] = self.variogram.covariance(np.array([h]))[0]

        # 4. Solve kriging system: C * weights = C0
        try:
            weights = np.linalg.solve(C, C0)
        except np.linalg.LinAlgError:
            # Singular matrix - return NaN
            result = {
                'estimate': np.nan,
                'variance': -1.0,  # Flag for singular matrix
            }
            if return_diagnostics:
                result.update({
                    'n_samples': n_neighbors,
                    'mean_distance': np.nan,
                    'slope': np.nan,
                    'efficiency': np.nan,
                })
            return result

        # 5. Calculate estimate
        # z*(x0) = mean + sum(wi * (zi - mean))
        residuals = neighbor_values - self.mean
        estimate = self.mean + np.sum(weights * residuals)

        # 6. Calculate kriging variance
        # sigma^2 = C(0,0) - sum(wi * C0i)
        C00 = self.variogram.covariance(np.array([0.0]))[0]  # Covariance at zero distance
        variance = C00 - np.sum(weights * C0)

        # Build result
        result = {
            'estimate': estimate,
            'variance': variance,
        }

        if return_diagnostics:
            # Calculate diagnostics
            mean_dist = np.mean(np.linalg.norm(neighbor_coords - point, axis=1))
            efficiency = 1.0 - variance / C00 if C00 > 0 else 0.0

            # Slope of regression (for OK this should be ~1)
            # For SK, we can compute sum of weights (should be close to 1 if data centered)
            slope = np.sum(weights)

            result.update({
                'n_samples': n_neighbors,
                'mean_distance': mean_dist,
                'slope': slope,
                'efficiency': efficiency,
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
        """
        Perform Ordinary Kriging at a single point.

        Ordinary Kriging system:
        [C   1] [w]   [C0]
        [1^T 0] [μ] = [1 ]

        Where:
        - C is the covariance matrix between data points (n x n)
        - w is the weights vector (n x 1)
        - μ is the Lagrange multiplier (unbiasedness constraint)
        - C0 is the covariance vector between data and estimation point (n x 1)

        Estimate: z*(x0) = sum(wi * zi)
        Variance: sigma^2 = C(0,0) - sum(wi * C0i) - μ
        """
        # 1. Find neighbors using isotropic search
        max_radius = self.search.radius[0] if len(self.search.radius) > 0 else self.search.radius

        # Query neighbors within search radius
        indices = self.kdtree.query_ball_point(point, r=max_radius)

        # Check minimum points requirement
        if len(indices) < self.search.min_points:
            result = {
                'estimate': np.nan,
                'variance': np.nan,
            }
            if return_diagnostics:
                result.update({
                    'n_samples': len(indices),
                    'mean_distance': np.nan,
                    'slope': np.nan,
                    'efficiency': np.nan,
                    'lagrange_multiplier': np.nan,
                })
            return result

        # Limit to max_points (use closest if too many)
        if len(indices) > self.search.max_points:
            # Calculate distances and sort
            dists = np.linalg.norm(self.data_coords[indices] - point, axis=1)
            sorted_idx = np.argsort(dists)
            indices = [indices[i] for i in sorted_idx[:self.search.max_points]]

        # Get neighbor coordinates and values
        neighbor_coords = self.data_coords[indices]
        neighbor_values = self.data_values[indices]
        n_neighbors = len(indices)

        # 2. Build covariance matrix C (between data points)
        C = np.zeros((n_neighbors, n_neighbors))
        for i in range(n_neighbors):
            for j in range(i, n_neighbors):
                # Calculate distance
                h = np.linalg.norm(neighbor_coords[i] - neighbor_coords[j])
                # Calculate covariance
                cov = self.variogram.covariance(np.array([h]))[0]
                C[i, j] = cov
                C[j, i] = cov  # Symmetric

        # 3. Build covariance vector C0 (between data and estimation point)
        C0 = np.zeros(n_neighbors)
        for i in range(n_neighbors):
            h = np.linalg.norm(neighbor_coords[i] - point)
            C0[i] = self.variogram.covariance(np.array([h]))[0]

        # 4. Build augmented system for Ordinary Kriging
        # [C   1] [w]   [C0]
        # [1^T 0] [μ] = [1 ]
        A = np.zeros((n_neighbors + 1, n_neighbors + 1))
        A[:n_neighbors, :n_neighbors] = C
        A[:n_neighbors, n_neighbors] = 1.0  # Last column
        A[n_neighbors, :n_neighbors] = 1.0  # Last row
        A[n_neighbors, n_neighbors] = 0.0   # Bottom-right corner

        b = np.zeros(n_neighbors + 1)
        b[:n_neighbors] = C0
        b[n_neighbors] = 1.0

        # 5. Solve kriging system
        try:
            solution = np.linalg.solve(A, b)
            weights = solution[:n_neighbors]
            lagrange = solution[n_neighbors]
        except np.linalg.LinAlgError:
            # Singular matrix - return NaN
            result = {
                'estimate': np.nan,
                'variance': -1.0,  # Flag for singular matrix
            }
            if return_diagnostics:
                result.update({
                    'n_samples': n_neighbors,
                    'mean_distance': np.nan,
                    'slope': np.nan,
                    'efficiency': np.nan,
                    'lagrange_multiplier': np.nan,
                })
            return result

        # 6. Calculate estimate
        # z*(x0) = sum(wi * zi)
        estimate = np.sum(weights * neighbor_values)

        # 7. Calculate kriging variance
        # sigma^2 = C(0,0) - sum(wi * C0i) - μ
        C00 = self.variogram.covariance(np.array([0.0]))[0]
        variance = C00 - np.sum(weights * C0) - lagrange

        # Build result
        result = {
            'estimate': estimate,
            'variance': variance,
        }

        if return_diagnostics:
            # Calculate diagnostics
            mean_dist = np.mean(np.linalg.norm(neighbor_coords - point, axis=1))
            efficiency = 1.0 - variance / C00 if C00 > 0 else 0.0

            # Slope of regression (sum of weights should be ~1 for OK)
            slope = np.sum(weights)

            result.update({
                'n_samples': n_neighbors,
                'mean_distance': mean_dist,
                'slope': slope,
                'efficiency': efficiency,
                'lagrange_multiplier': lagrange,
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
