"""Interior profiles of rocky planets compatible with a measured mass and radius.

Two-layer planets (Fe core under an MgSiO3 mantle) in hydrostatic equilibrium on an
adiabat, with material properties interpolated from the PALEOS lookup tables.
"""

__version__ = '0.1.0'

from .structure import NoSolutionError, integrate
from .tables import load_tables

__all__ = ['NoSolutionError', 'integrate', 'load_tables']
