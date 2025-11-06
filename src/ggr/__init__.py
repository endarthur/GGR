"""
GGR - Geostatistical Grid Realization

Modern geostatistics in Python - kriging and simulation for the browser and beyond.
"""

__version__ = "0.1.0"

# Main functional API
from .kriging import krige, simple_krige, ordinary_krige

# Class-based API
from .kriging import SimpleKrige, OrdinaryKrige

# Utilities
from .variogram import Variogram
from .search import AnisotropicSearch
from .grid import Grid

__all__ = [
    # Functional API
    "krige",
    "simple_krige", 
    "ordinary_krige",
    # Class API
    "SimpleKrige",
    "OrdinaryKrige",
    # Utilities
    "Variogram",
    "AnisotropicSearch",
    "Grid",
]
