"""
Variogram models and utilities.

This module provides variogram model definitions that can be used
with kriging algorithms. It accepts both dict-based and object-based
variogram specifications for flexibility.
"""

from typing import Union, Dict, Literal
import numpy as np
from numpy.typing import NDArray


class Variogram:
    """
    Variogram model specification.
    
    Stores variogram parameters and provides covariance calculations.
    Can be constructed from dict or parameters.
    
    Parameters
    ----------
    model : {'spherical', 'exponential', 'gaussian', 'linear', 'power'}
        Variogram model type
    range_ : float or array-like
        Range parameter(s). Can be scalar or [major, semi-major, minor]
    sill : float
        Sill (total variance)
    nugget : float, optional
        Nugget effect (default: 0.0)
    angles : tuple of float, optional
        (dip_direction, dip, rake) in degrees for anisotropy
    
    Examples
    --------
    >>> vario = Variogram('spherical', range_=500, sill=1.0, nugget=0.1)
    >>> vario = Variogram.from_dict({'model': 'spherical', 'range': 500, ...})
    """
    
    def __init__(
        self,
        model: Literal['spherical', 'exponential', 'gaussian', 'linear', 'power'],
        range_: Union[float, NDArray],
        sill: float,
        nugget: float = 0.0,
        angles: tuple = (0.0, 0.0, 0.0),
    ):
        self.model = model
        self.range = np.atleast_1d(range_)
        self.sill = sill
        self.nugget = nugget
        self.angles = angles  # (dip_direction, dip, rake)
        
        # Validate
        self._validate()
    
    def _validate(self) -> None:
        """Validate variogram parameters."""
        if self.sill <= 0:
            raise ValueError("Sill must be positive")
        if self.nugget < 0:
            raise ValueError("Nugget must be non-negative")
        if np.any(self.range <= 0):
            raise ValueError("Range must be positive")
    
    @classmethod
    def from_dict(cls, params: Dict) -> 'Variogram':
        """
        Create Variogram from dictionary.
        
        Parameters
        ----------
        params : dict
            Dictionary with keys: 'model', 'range', 'sill', 'nugget' (optional)
        
        Returns
        -------
        Variogram
            Variogram instance
        """
        return cls(
            model=params['model'],
            range_=params.get('range', params.get('range_')),
            sill=params['sill'],
            nugget=params.get('nugget', 0.0),
            angles=params.get('angles', (0.0, 0.0, 0.0)),
        )
    
    def covariance(self, h: NDArray) -> NDArray:
        """
        Calculate covariance for given distances.
        
        Parameters
        ----------
        h : ndarray
            Distances
        
        Returns
        -------
        ndarray
            Covariance values
        """
        # TODO: Implement covariance calculation for each model type
        raise NotImplementedError("Covariance calculation not yet implemented")
    
    def __repr__(self) -> str:
        return (
            f"Variogram(model='{self.model}', range={self.range}, "
            f"sill={self.sill}, nugget={self.nugget})"
        )


def parse_variogram(variogram: Union[Variogram, Dict]) -> Variogram:
    """
    Parse variogram input into Variogram object.
    
    Accepts either a Variogram instance or a dictionary specification.
    
    Parameters
    ----------
    variogram : Variogram or dict
        Variogram specification
    
    Returns
    -------
    Variogram
        Variogram object
    """
    if isinstance(variogram, Variogram):
        return variogram
    elif isinstance(variogram, dict):
        return Variogram.from_dict(variogram)
    else:
        raise TypeError(
            f"Variogram must be Variogram instance or dict, got {type(variogram)}"
        )
