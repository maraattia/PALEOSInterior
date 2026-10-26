"""PALEOS lookup tables: download, cache, and interpolation.

The tables are log-uniform grids in pressure and temperature with the columns
P, T, rho, u, s, C_P, C_V, alpha, nabla_ad, phase (SI units). Properties are
interpolated bilinearly in (log10 P, log10 T), as the table headers prescribe.
"""

import hashlib
import math
import os
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    clamped to its edges.
    """

    def __init__(self, path):
        self.path = Path(path)
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
        # Plain nested lists make scalar lookups in the ODE right-hand side fast.
        self._rho = self.grids['rho'].tolist()
        self._nabla = self.grids['nabla_ad'].tolist()

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
        out = []
        for g in (self._rho, self._nabla):
            lo = g[i][j] + fy * (g[i][j + 1] - g[i][j])
            hi = g[i + 1][j] + fy * (g[i + 1][j + 1] - g[i + 1][j])
            out.append(lo + fx * (hi - lo))
        return out[0], out[1]

    def evaluate(self, P, T):
        """Return a dict of property arrays and phase labels on arrays of (P, T)."""
        x = (np.log10(P) - self._p0) / self._step_p
        y = (np.log10(T) - self._t0) / self._step_t
        x = np.clip(x, 0.0, self._n_p - 1.0)
        y = np.clip(y, 0.0, self._n_t - 1.0)
        i = np.minimum(x.astype(int), self._n_p - 2)
        j = np.minimum(y.astype(int), self._n_t - 2)
        fx, fy = x - i, y - j
        out = {}
        for key, g in self.grids.items():
            lo = g[i, j] + fy * (g[i, j + 1] - g[i, j])
            hi = g[i + 1, j] + fy * (g[i + 1, j + 1] - g[i + 1, j])
            out[key] = lo + fx * (hi - lo)
        codes = self._phase_grid[np.rint(x).astype(int), np.rint(y).astype(int)]
        out['phase'] = np.where(codes >= 0, self.phase_names[codes], '')
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
    return Tables(MaterialTable(paths['iron']), MaterialTable(paths['mgsio3']), checksums)
