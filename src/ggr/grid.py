"""
Grid definition and utilities.

Provides tools for defining regular grids and arbitrary point sets
for kriging estimation.
"""

from typing import Union, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import pandas as pd


class Grid:
    """
    Grid definition for kriging estimation.
    
    Can represent either a regular grid or arbitrary points.
    
    Parameters
    ----------
    For regular grid:
        xmin, xmax, nx : float, float, int
            X dimension definition
        ymin, ymax, ny : float, float, int
            Y dimension definition
        zmin, zmax, nz : float, float, int, optional
            Z dimension definition (for 3D)
    
    For arbitrary points:
        points : ndarray
            Array of coordinates, shape (n, 2) or (n, 3)
    
    Examples
    --------
    >>> # Regular 2D grid
    >>> grid = Grid(xmin=0, xmax=1000, nx=100,
    ...             ymin=0, ymax=1000, ny=100)
    
    >>> # Regular 3D grid
    >>> grid = Grid(xmin=0, xmax=1000, nx=100,
    ...             ymin=0, ymax=1000, ny=100,
    ...             zmin=0, zmax=100, nz=10)
    
    >>> # Arbitrary points
    >>> points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
    >>> grid = Grid(points=points)
    
    >>> # From dict
    >>> grid = Grid.from_dict({'xmin': 0, 'xmax': 1000, 'nx': 100, ...})
    """
    
    def __init__(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        nx: Optional[int] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        ny: Optional[int] = None,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        nz: Optional[int] = None,
        points: Optional[NDArray] = None,
    ):
        if points is not None:
            # Arbitrary points
            self.points = np.asarray(points)
            self.ndim = self.points.shape[1]
            self.is_regular = False
            self.n_points = len(self.points)
        else:
            # Regular grid
            self.is_regular = True
            
            # Check for 2D vs 3D
            if zmin is None and zmax is None and nz is None:
                # 2D grid
                self.ndim = 2
                self.xmin, self.xmax, self.nx = xmin, xmax, nx
                self.ymin, self.ymax, self.ny = ymin, ymax, ny
                self.n_points = nx * ny
            else:
                # 3D grid
                self.ndim = 3
                self.xmin, self.xmax, self.nx = xmin, xmax, nx
                self.ymin, self.ymax, self.ny = ymin, ymax, ny
                self.zmin, self.zmax, self.nz = zmin, zmax, nz
                self.n_points = nx * ny * nz
            
            # Generate grid points
            self.points = self._generate_regular_grid()
        
        # Validate
        self._validate()
    
    def _validate(self) -> None:
        """Validate grid parameters."""
        if self.points is None or len(self.points) == 0:
            raise ValueError("Grid must have points")
        if self.ndim not in [2, 3]:
            raise ValueError("Grid must be 2D or 3D")
    
    def _generate_regular_grid(self) -> NDArray:
        """Generate regular grid points."""
        if self.ndim == 2:
            x = np.linspace(self.xmin, self.xmax, self.nx)
            y = np.linspace(self.ymin, self.ymax, self.ny)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            points = np.column_stack([xx.ravel(), yy.ravel()])
        else:  # 3D
            x = np.linspace(self.xmin, self.xmax, self.nx)
            y = np.linspace(self.ymin, self.ymax, self.ny)
            z = np.linspace(self.zmin, self.zmax, self.nz)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        return points
    
    @classmethod
    def from_dict(cls, params: Dict) -> 'Grid':
        """
        Create Grid from dictionary.
        
        Parameters
        ----------
        params : dict
            Dictionary with grid parameters
        
        Returns
        -------
        Grid
            Grid instance
        """
        return cls(
            xmin=params.get('xmin'),
            xmax=params.get('xmax'),
            nx=params.get('nx'),
            ymin=params.get('ymin'),
            ymax=params.get('ymax'),
            ny=params.get('ny'),
            zmin=params.get('zmin'),
            zmax=params.get('zmax'),
            nz=params.get('nz'),
            points=params.get('points'),
        )
    
    def get_shape(self) -> tuple:
        """Get grid shape for regular grids."""
        if not self.is_regular:
            raise ValueError("Shape only defined for regular grids")
        if self.ndim == 2:
            return (self.nx, self.ny)
        else:
            return (self.nx, self.ny, self.nz)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert grid points to DataFrame."""
        if self.ndim == 2:
            return pd.DataFrame(self.points, columns=['x', 'y'])
        else:
            return pd.DataFrame(self.points, columns=['x', 'y', 'z'])
    
    def __repr__(self) -> str:
        if self.is_regular:
            if self.ndim == 2:
                return f"Grid(nx={self.nx}, ny={self.ny}, n_points={self.n_points})"
            else:
                return f"Grid(nx={self.nx}, ny={self.ny}, nz={self.nz}, n_points={self.n_points})"
        else:
            return f"Grid(arbitrary_points, n_points={self.n_points}, ndim={self.ndim})"


def parse_grid(grid: Union[Grid, Dict, NDArray]) -> Grid:
    """
    Parse grid input into Grid object.
    
    Parameters
    ----------
    grid : Grid, dict, or ndarray
        Grid specification
    
    Returns
    -------
    Grid
        Grid object
    """
    if isinstance(grid, Grid):
        return grid
    elif isinstance(grid, dict):
        return Grid.from_dict(grid)
    elif isinstance(grid, np.ndarray):
        return Grid(points=grid)
    else:
        raise TypeError(f"Grid must be Grid, dict, or ndarray, got {type(grid)}")
