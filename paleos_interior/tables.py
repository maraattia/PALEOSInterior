"""PALEOS lookup tables: download, cache, and interpolation.

The tables are log-uniform grids in pressure and temperature with the columns
P, T, rho, u, s, C_P, C_V, alpha, nabla_ad, phase (SI units). Properties are
interpolated bilinearly in (log10 P, log10 T), as the table headers prescribe.

Near the melting curve each phase is continued a few nodes into the other along
every pressure row, by linear extrapolation in log10 T, so that solid and liquid
properties can each be interpolated without mixing in nodes of the other phase.
The result is (1 - s) times the solid value plus s times the liquid value, with s
rising linearly in log10 T from 0 at log10 T_m - MELT_BLEND to 1 at
log10 T_m + MELT_BLEND, T_m being the melting temperature. Solid and liquid
properties are averaged only inside that band, which keeps the structure equations
continuous where an adiabat follows the melting curve,
and away from the curve the interpolation is plain bilinear. The blend applies to rho
and nabla_ad. C_P, C_V, and alpha, which enter the bulk moduli nonlinearly, take the
interpolant of the point's own phase, so the moduli of the two phases are never mixed.
"""

import hashlib
import math
import os
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import MELT_BLEND
from .rheology import melting_temperature

RELEASE = {
    'version': '1.3.0',
    'doi': '10.5281/zenodo.22776069',
    'url': 'https://zenodo.org/api/records/22776069/files/{name}/content',
    'files': {
        'iron': ('paleos_iron_eos_table_pt.dat', '259a2267dcf58af3b2c4dd389105b597'),
        'mgsio3': ('paleos_mgsio3_eos_table_pt.dat', '1f75d518bf192e66b94ec0bda190ada7'),
    },
}

PROPERTIES = ('rho', 'cp', 'cv', 'alpha', 'nabla_ad')
_COLUMNS = {'rho': 2, 'cp': 5, 'cv': 6, 'alpha': 7, 'nabla_ad': 8}
_EXTEND = 3  # nodes each phase is continued past the melting curve
_BLENDED = ('rho', 'nabla_ad')


def cache_dir():
    """Return the directory where downloaded tables are kept."""
    if 'PALEOS_INTERIOR_CACHE' in os.environ:
        return Path(os.environ['PALEOS_INTERIOR_CACHE'])
    base = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(base) / 'paleos_interior' / RELEASE['version']


def _md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _fetch(name, md5):
    """Return the cached path of a release file, downloading it if needed."""
    path = cache_dir() / name
    if path.exists() and _md5(path) == md5:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.part')
    print(f'Downloading {name} from Zenodo (PALEOS tables {RELEASE["version"]})')
    urllib.request.urlretrieve(RELEASE['url'].format(name=name), partial)
    if _md5(partial) != md5:
        partial.unlink()
        raise OSError(f'Checksum mismatch for {name} downloaded from Zenodo')
    partial.replace(path)
    return path


class MaterialTable:
    """One PALEOS lookup table on its regular (log10 P, log10 T) grid.

    Grid points the table omits (non-convergent MgSiO3 points) are NaN, and any
    interpolation touching them returns NaN. Queries outside the grid are
    clamped to its edges. ``material`` is 'iron' or 'mgsio3'.
    """

    def __init__(self, path, material):
        self.path = Path(path)
        self.material = material
        data = np.loadtxt(path, comments='#', usecols=range(9))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            phase = np.loadtxt(path, comments='#', usecols=9, dtype=str)

        log_p, log_t = np.log10(data[:, 0]), np.log10(data[:, 1])
        axis_p, axis_t = np.unique(log_p), np.unique(log_t)
        self._step_p = _uniform_step(axis_p)
        self._step_t = _uniform_step(axis_t)
        self._p0, self._t0 = axis_p[0], axis_t[0]
        self._n_p, self._n_t = len(axis_p), len(axis_t)
        self.log_p_range = (axis_p[0], axis_p[-1])
        self.log_t_range = (axis_t[0], axis_t[-1])

        i = np.rint((log_p - self._p0) / self._step_p).astype(int)
        j = np.rint((log_t - self._t0) / self._step_t).astype(int)
        self.grids = {}
        for key, col in _COLUMNS.items():
            grid = np.full((self._n_p, self._n_t), np.nan)
            grid[i, j] = data[:, col]
            self.grids[key] = grid

        self.phase_names, codes = np.unique(phase, return_inverse=True)
        self._phase_grid = np.full((self._n_p, self._n_t), -1, dtype=int)
        self._phase_grid[i, j] = codes
        # Node class: 1 liquid, 0 solid, -1 missing. Every pressure row holds solid
        # nodes below liquid ones, and the labels agree with the melting curve.
        is_liquid = np.isin(self._phase_grid, np.flatnonzero(self.phase_names == 'liquid'))
        node_class = np.where(self._phase_grid < 0, -1, is_liquid.astype(int))
        self.solid_grids = {k: _extend(g, node_class, 0) for k, g in self.grids.items()}
        self.liquid_grids = {k: _extend(g, node_class, 1) for k, g in self.grids.items()}
        both = np.isfinite(self.solid_grids['rho']) & np.isfinite(self.liquid_grids['rho'])
        self._near_melt = both[:-1, :-1] & both[:-1, 1:] & both[1:, :-1] & both[1:, 1:]
        self._log_tm_grid = np.log10(melting_temperature(material, 10.0**axis_p))

        # Plain nested lists make scalar lookups in the ODE right-hand side fast.
        self._rho = self.grids['rho'].tolist()
        self._nabla = self.grids['nabla_ad'].tolist()
        self._rho_s = self.solid_grids['rho'].tolist()
        self._nabla_s = self.solid_grids['nabla_ad'].tolist()
        self._rho_l = self.liquid_grids['rho'].tolist()
        self._nabla_l = self.liquid_grids['nabla_ad'].tolist()
        self._near = self._near_melt.tolist()
        self._log_tm = self._log_tm_grid.tolist()

    def _cell(self, P, T):
        x = (math.log10(P) - self._p0) / self._step_p
        y = (math.log10(T) - self._t0) / self._step_t
        x = min(max(x, 0.0), self._n_p - 1.0)
        y = min(max(y, 0.0), self._n_t - 1.0)
        i = min(int(x), self._n_p - 2)
        j = min(int(y), self._n_t - 2)
        return i, j, x - i, y - j

    def rho_nabla(self, P, T):
        """Return (density, adiabatic gradient) at one (P, T) point."""
        i, j, fx, fy = self._cell(P, T)
        if not self._near[i][j]:
            return _bilinear(self._rho, i, j, fx, fy), _bilinear(self._nabla, i, j, fx, fy)
        log_tm = self._log_tm[i] + fx * (self._log_tm[i + 1] - self._log_tm[i])
        s = min(max(0.5 + (math.log10(T) - log_tm) / (2.0 * MELT_BLEND), 0.0), 1.0)
        rho = nabla = 0.0
        if s < 1.0:
            rho += (1.0 - s) * _bilinear(self._rho_s, i, j, fx, fy)
            nabla += (1.0 - s) * _bilinear(self._nabla_s, i, j, fx, fy)
        if s > 0.0:
            rho += s * _bilinear(self._rho_l, i, j, fx, fy)
            nabla += s * _bilinear(self._nabla_l, i, j, fx, fy)
        return rho, nabla

    def evaluate(self, P, T):
        """Return a dict of property arrays and phase labels on arrays of (P, T)."""
        x = (np.log10(P) - self._p0) / self._step_p
        y = (np.log10(T) - self._t0) / self._step_t
        x = np.clip(x, 0.0, self._n_p - 1.0)
        y = np.clip(y, 0.0, self._n_t - 1.0)
        i = np.minimum(x.astype(int), self._n_p - 2)
        j = np.minimum(y.astype(int), self._n_t - 2)
        fx, fy = x - i, y - j
        near = self._near_melt[i, j]
        tm = self._log_tm_grid
        log_tm = tm[i] + fx * (tm[i + 1] - tm[i])
        s = np.clip(0.5 + (np.log10(T) - log_tm) / (2.0 * MELT_BLEND), 0.0, 1.0)
        s = s[near]
        out = {}
        for key, g in self.grids.items():
            out[key] = _bilinear(g, i, j, fx, fy)
            if near.any():
                args = (i[near], j[near], fx[near], fy[near])
                solid = _bilinear(self.solid_grids[key], *args)
                liquid = _bilinear(self.liquid_grids[key], *args)
                if key in _BLENDED:
                    solid = np.where(s < 1.0, solid, 0.0)
                    liquid = np.where(s > 0.0, liquid, 0.0)
                    out[key][near] = (1.0 - s) * solid + s * liquid
                else:
                    out[key][near] = np.where(s > 0.5, liquid, solid)
        codes = self._phase_grid[np.rint(x).astype(int), np.rint(y).astype(int)]
        out['phase'] = np.where(codes >= 0, self.phase_names[codes], '').astype('<U32')
        return out

    def solid_phase(self, P, T):
        """Return the label of the nearest solid node at or below T on the row of P."""
        i = np.rint((np.log10(P) - self._p0) / self._step_p).astype(int)
        j = np.rint((np.log10(T) - self._t0) / self._step_t).astype(int)
        i = np.clip(i, 0, self._n_p - 1)
        j = np.clip(j, 0, self._n_t - 1)
        names = list(self.phase_names)
        out = []
        for ii, jj in zip(i, j):
            row = self._phase_grid[ii, : jj + 1][::-1]
            solid = [c for c in row if c >= 0 and names[c] != 'liquid']
            out.append(names[solid[0]] if solid else '')
        return np.array(out, dtype='<U32')


def _bilinear(g, i, j, fx, fy):
    """Bilinear interpolation on grid g (nested list or array) in cell (i, j)."""
    if isinstance(g, list):
        lo = g[i][j] + fy * (g[i][j + 1] - g[i][j])
        hi = g[i + 1][j] + fy * (g[i + 1][j + 1] - g[i + 1][j])
    else:
        lo = g[i, j] + fy * (g[i, j + 1] - g[i, j])
        hi = g[i + 1, j] + fy * (g[i + 1, j + 1] - g[i + 1, j])
    return lo + fx * (hi - lo)


def _extend(grid, node_class, phase):
    """Return grid restricted to one phase (0 solid, 1 liquid) and continued past it.

    Along every pressure row, the _EXTEND nodes beyond the last node of the phase
    take the linear extrapolation in log10 T of its two outermost nodes. All other
    nodes of the other phase, and missing nodes, are NaN.
    """
    out = np.where(node_class == phase, grid, np.nan)
    n_t = grid.shape[1]
    for i, row in enumerate(node_class):
        own = np.flatnonzero(row == phase)
        if not own.size or not (row == 1 - phase).any():
            continue
        # The solid side is continued upward in T, the liquid side downward.
        edge, step = (own.max(), 1) if phase == 0 else (own.min(), -1)
        inner = edge - step
        if not 0 <= inner < n_t or row[inner] != phase:
            continue
        slope = grid[i, edge] - grid[i, inner]
        for k in range(1, _EXTEND + 1):
            if 0 <= edge + k * step < n_t:
                out[i, edge + k * step] = grid[i, edge] + k * slope
    return out


def _uniform_step(axis):
    steps = np.diff(axis)
    # Axes are printed with finite digits, so allow a small spread in the step.
    if np.ptp(steps) > 1e-4 * steps.mean():
        raise ValueError('Table axis is not log-uniform')
    return (axis[-1] - axis[0]) / (len(axis) - 1)


@dataclass
class Tables:
    """The iron and MgSiO3 tables used to build a planet, with their provenance."""

    iron: MaterialTable
    mgsio3: MaterialTable
    checksums: dict

    @property
    def is_release(self):
        """True when both files match the pinned Zenodo release."""
        return all(self.checksums[k] == md5 for k, (_, md5) in RELEASE['files'].items())

    def provenance(self):
        """Return header lines naming the table release and file checksums."""
        if self.is_release:
            lines = [f'PALEOS tables {RELEASE["version"]}, doi:{RELEASE["doi"]}']
        else:
            lines = ['PALEOS tables from local files that differ from a Zenodo release']
        for key in ('iron', 'mgsio3'):
            name = getattr(self, key).path.name
            lines.append(f'{name} md5 {self.checksums[key]}')
        return lines


def load_tables(table_dir=None):
    """Load the PALEOS iron and MgSiO3 tables.

    With ``table_dir=None`` the pinned Zenodo release is downloaded on first use
    and cached (see ``cache_dir``). Otherwise the two files are read from
    ``table_dir`` under their release names.
    """
    paths = {}
    for material, (name, md5) in RELEASE['files'].items():
        if table_dir is None:
            paths[material] = _fetch(name, md5)
        else:
            paths[material] = Path(table_dir).expanduser() / name
    checksums = {k: _md5(path) for k, path in paths.items()}
    return Tables(
        MaterialTable(paths['iron'], 'iron'),
        MaterialTable(paths['mgsio3'], 'mgsio3'),
        checksums,
    )
