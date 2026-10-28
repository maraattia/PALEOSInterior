"""Radial profile of one planet and the quantities derived from it."""

from dataclasses import dataclass, field

import numpy as np

from .constants import G

COLUMNS = (
    ('r', 'm'),
    ('m', 'kg'),
    ('g', 'm s-2'),
    ('P', 'Pa'),
    ('T', 'K'),
    ('rho', 'kg m-3'),
    ('cp', 'J kg-1 K-1'),
    ('cv', 'J kg-1 K-1'),
    ('alpha', 'K-1'),
    ('nabla_ad', '1'),
    ('K_T', 'Pa'),
    ('K_S', 'Pa'),
    ('V_phi', 'm s-1'),
    ('mu', 'Pa'),
    ('V_P', 'm s-1'),
    ('V_S', 'm s-1'),
    ('eta', 'Pa s'),
    ('layer', ''),
    ('phase', ''),
)


@dataclass
class Profile:
    """Interior structure of one planet, ordered from the surface to the center.

    Array attributes follow ``COLUMNS``. The core-mantle boundary radius appears
    twice, once with mantle and once with core properties, so the density jump
    is resolved.
    """

    mass: float
    radius: float
    cmf: float
    T_pot: float
    dT_cmb: float
    columns: dict = field(repr=False)
    grain_size: float = np.nan
    R_cmb: float = np.nan
    P_cmb: float = np.nan
    T_cmb: float = np.nan

    def __getattr__(self, name):
        columns = self.__dict__.get('columns', {})
        if name in columns:
            return columns[name]
        raise AttributeError(name)

    @property
    def P_center(self):
        return self.columns['P'][-1]

    @property
    def T_center(self):
        return self.columns['T'][-1]

    def scalars(self):
        """Return the per-planet scalars in SI units."""
        return {
            'M': self.mass,
            'R': self.radius,
            'CMF': self.cmf,
            'T_pot': self.T_pot,
            'dT_cmb': self.dT_cmb,
            'grain_size': self.grain_size,
            'R_cmb': self.R_cmb,
            'P_cmb': self.P_cmb,
            'T_cmb': self.T_cmb,
            'P_inner': self.P_center,
            'T_inner': self.T_center,
        }


def derived_columns(r, m, props):
    """Return gravity, the bulk moduli, and the bulk, P, and S wave speeds.

    K_T follows from C_P - C_V = T alpha^2 K_T / rho and K_S = K_T C_P / C_V,
    so both inherit any error in the tabulated heat capacities. ``props['mu']``
    is the unrelaxed shear modulus. V_P and V_S use the relaxed one, which is
    zero in liquids because their Maxwell times are of order 1e-11 s or less, so
    there V_S = 0 and V_P = V_phi.
    """
    rho, cp, cv, alpha = props['rho'], props['cp'], props['cv'], props['alpha']
    T = props['T']
    mu = np.where(props['phase'] == 'liquid', 0.0, props['mu'])
    with np.errstate(divide='ignore', invalid='ignore'):
        K_T = rho * (cp - cv) / (T * alpha**2)
        K_S = K_T * cp / cv
        V_phi = np.sqrt(K_S / rho)
        V_P = np.sqrt((K_S + 4.0 * mu / 3.0) / rho)
        V_S = np.sqrt(mu / rho)
    return {
        'g': G * m / r**2,
        'K_T': K_T,
        'K_S': K_S,
        'V_phi': V_phi,
        'V_P': V_P,
        'V_S': V_S,
    }
