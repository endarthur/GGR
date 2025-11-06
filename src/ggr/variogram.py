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
        valid_models = ['spherical', 'exponential', 'gaussian', 'linear', 'power']
        if self.model not in valid_models:
            raise ValueError(
                f"Model must be one of {valid_models}, got '{self.model}'"
            )
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

        Covariance is calculated as C(h) = sill - gamma(h),
        where gamma(h) is the variogram value at distance h.

        Parameters
        ----------
        h : ndarray
            Distances (scalar or array)

        Returns
        -------
        ndarray
            Covariance values

        Notes
        -----
        At h=0, returns sill (not sill-nugget). The nugget effect
        is included in the variogram calculation for h>0.

        Variogram models:
        - Spherical: gamma(h) = nugget + (sill-nugget) * [1.5*(h/a) - 0.5*(h/a)^3] for h<a
                                       = sill for h>=a
        - Exponential: gamma(h) = nugget + (sill-nugget) * [1 - exp(-h/a)]
        - Gaussian: gamma(h) = nugget + (sill-nugget) * [1 - exp(-(h/a)^2)]
        """
        h = np.atleast_1d(h).astype(float)

        # At h=0, covariance = sill
        gamma = np.zeros_like(h)

        # For h > 0, calculate variogram value based on model
        nonzero = h > 0

        if self.model == 'spherical':
            gamma[nonzero] = self._spherical_variogram(h[nonzero])
        elif self.model == 'exponential':
            gamma[nonzero] = self._exponential_variogram(h[nonzero])
        elif self.model == 'gaussian':
            gamma[nonzero] = self._gaussian_variogram(h[nonzero])
        else:
            raise ValueError(f"Model '{self.model}' not yet implemented")

        # Covariance = sill - gamma
        return self.sill - gamma

    def _spherical_variogram(self, h: NDArray) -> NDArray:
        """
        Calculate spherical variogram values.

        Parameters
        ----------
        h : ndarray
            Distances (must be > 0)

        Returns
        -------
        ndarray
            Variogram values
        """
        # Use the first range value (isotropic case)
        a = self.range[0]
        h_norm = h / a

        # gamma = nugget + (sill-nugget) * f(h)
        # where f(h) = 1.5*(h/a) - 0.5*(h/a)^3 for h < a
        #            = 1.0 for h >= a
        gamma = np.full_like(h, self.sill)  # h >= a case

        within_range = h < a
        if np.any(within_range):
            h_norm_wr = h_norm[within_range]
            f_h = 1.5 * h_norm_wr - 0.5 * h_norm_wr**3
            gamma[within_range] = self.nugget + (self.sill - self.nugget) * f_h

        return gamma

    def _exponential_variogram(self, h: NDArray) -> NDArray:
        """
        Calculate exponential variogram values.

        Parameters
        ----------
        h : ndarray
            Distances (must be > 0)

        Returns
        -------
        ndarray
            Variogram values
        """
        # Use the first range value (isotropic case)
        a = self.range[0]

        # gamma = nugget + (sill-nugget) * [1 - exp(-h/a)]
        f_h = 1.0 - np.exp(-h / a)
        gamma = self.nugget + (self.sill - self.nugget) * f_h

        return gamma

    def _gaussian_variogram(self, h: NDArray) -> NDArray:
        """
        Calculate gaussian variogram values.

        Parameters
        ----------
        h : ndarray
            Distances (must be > 0)

        Returns
        -------
        ndarray
            Variogram values
        """
        # Use the first range value (isotropic case)
        a = self.range[0]

        # gamma = nugget + (sill-nugget) * [1 - exp(-(h/a)^2)]
        f_h = 1.0 - np.exp(-(h / a)**2)
        gamma = self.nugget + (self.sill - self.nugget) * f_h

        return gamma
    
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
