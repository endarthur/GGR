"""
Search strategies for kriging.

Provides tools for defining and applying anisotropic search ellipsoids
and octant search strategies.
"""

from typing import Union, Tuple, Literal, Optional
import numpy as np
from numpy.typing import NDArray


class AnisotropicSearch:
    """
    Anisotropic search ellipsoid definition.
    
    Defines an oriented ellipsoidal search volume with optional octant
    subdivision for kriging neighborhood selection.
    
    Parameters
    ----------
    radius : float or array-like
        Search radius. Scalar for isotropic, or [major, semi-major, minor]
    dip_direction : float, optional
        Dip direction in degrees (default convention)
    dip : float, optional
        Dip in degrees
    rake : float, optional
        Rake in degrees
    convention : {'dip_dir_dip_rake', 'gslib', 'strike_dip'}, optional
        Angle convention (default: 'dip_dir_dip_rake')
    octants : bool, optional
        Use octant search (default: False)
    max_points : int, optional
        Maximum points to use (default: 50)
    min_points : int, optional
        Minimum points required (default: 4)
    
    Examples
    --------
    >>> # Isotropic search
    >>> search = AnisotropicSearch(radius=500)
    
    >>> # Anisotropic with octants
    >>> search = AnisotropicSearch(
    ...     radius=[500, 300, 100],
    ...     dip_direction=45, dip=30, rake=0,
    ...     octants=True
    ... )
    """
    
    def __init__(
        self,
        radius: Union[float, NDArray],
        dip_direction: float = 0.0,
        dip: float = 0.0,
        rake: float = 0.0,
        convention: Literal['dip_dir_dip_rake', 'gslib', 'strike_dip'] = 'dip_dir_dip_rake',
        octants: bool = False,
        max_points: int = 50,
        min_points: int = 4,
    ):
        self.radius = np.atleast_1d(radius)
        self.octants = octants
        self.max_points = max_points
        self.min_points = min_points
        
        # Store original angles
        self._dip_direction = dip_direction
        self._dip = dip
        self._rake = rake
        self._convention = convention
        
        # Convert to standard convention (dip_direction, dip, rake)
        self.dip_direction, self.dip, self.rake = self._convert_angles(
            dip_direction, dip, rake, convention
        )
        
        # Create rotation matrix
        self.rotation_matrix = self._build_rotation_matrix()
        
        # Validate
        self._validate()
    
    def _validate(self) -> None:
        """Validate search parameters."""
        if np.any(self.radius <= 0):
            raise ValueError("Search radius must be positive")
        if self.max_points < self.min_points:
            raise ValueError("max_points must be >= min_points")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
    
    @staticmethod
    def _convert_angles(
        angle1: float,
        angle2: float, 
        angle3: float,
        convention: str
    ) -> Tuple[float, float, float]:
        """
        Convert from various angle conventions to dip_direction/dip/rake.
        
        Parameters
        ----------
        angle1, angle2, angle3 : float
            Angles in degrees
        convention : str
            Convention name
        
        Returns
        -------
        tuple of float
            (dip_direction, dip, rake) in standard convention
        """
        if convention == 'dip_dir_dip_rake':
            return angle1, angle2, angle3
        
        elif convention == 'gslib':
            # GSLIB uses azimuth, dip, plunge
            azimuth, dip, plunge = angle1, angle2, angle3
            # Convert azimuth to dip direction
            dip_direction = (90 - azimuth) % 360
            rake = plunge
            return dip_direction, dip, rake
        
        elif convention == 'strike_dip':
            # Strike-dip-rake convention
            strike, dip, rake = angle1, angle2, angle3
            # Dip direction is strike + 90 (right-hand rule)
            dip_direction = (strike + 90) % 360
            return dip_direction, dip, rake
        
        else:
            raise ValueError(f"Unknown angle convention: {convention}")
    
    def _build_rotation_matrix(self) -> NDArray:
        """
        Build 3D rotation matrix from dip_direction/dip/rake.
        
        Returns
        -------
        ndarray
            3x3 rotation matrix
        """
        # Convert to radians
        dd_rad = np.radians(self.dip_direction)
        dip_rad = np.radians(self.dip)
        rake_rad = np.radians(self.rake)
        
        # Build rotation matrix (ZXZ convention)
        # TODO: Implement proper 3D rotation matrix construction
        # This is a placeholder
        cos_dd, sin_dd = np.cos(dd_rad), np.sin(dd_rad)
        cos_dip, sin_dip = np.cos(dip_rad), np.sin(dip_rad)
        cos_rake, sin_rake = np.cos(rake_rad), np.sin(rake_rad)
        
        # Simplified version - needs proper implementation
        R = np.eye(3)
        return R
    
    def transform_coordinates(self, coords: NDArray, center: NDArray) -> NDArray:
        """
        Transform coordinates into search ellipsoid space.
        
        Parameters
        ----------
        coords : ndarray, shape (n, 3)
            Coordinates to transform
        center : ndarray, shape (3,)
            Center point of search
        
        Returns
        -------
        ndarray
            Transformed coordinates
        """
        # Translate to center
        translated = coords - center
        
        # Rotate
        rotated = translated @ self.rotation_matrix.T
        
        # Scale by radius
        if len(self.radius) == 3:
            scaled = rotated / self.radius
        else:
            scaled = rotated / self.radius[0]
        
        return scaled
    
    def is_isotropic(self) -> bool:
        """Check if search is isotropic."""
        if len(self.radius) == 1:
            return True
        return np.allclose(self.radius, self.radius[0])
    
    def __repr__(self) -> str:
        return (
            f"AnisotropicSearch(radius={self.radius}, "
            f"dd={self.dip_direction:.1f}, dip={self.dip:.1f}, rake={self.rake:.1f}, "
            f"octants={self.octants}, max_points={self.max_points})"
        )
