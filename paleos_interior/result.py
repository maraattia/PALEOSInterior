"""Interior profiles compatible with a measured mass and radius, and their CSV output."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .constants import GRAIN_SIZE, M_EARTH, P_SURFACE, R_EARTH, R_INNER
from .line import _uncertain, compatible_envelope, compatible_line
from .profile import COLUMNS
from .rheology import SOURCES
from .tables import load_tables

LINE_COLUMNS = (
    ('T_pot', 'K'),
    ('CMF', '1'),
    ('R_cmb', 'm'),
    ('P_cmb', 'Pa'),
    ('T_cmb', 'K'),
)
SUMMARY_COLUMNS = (
    ('profile', ''),
    ('M', 'kg'),
    ('R', 'm'),
    ('R_rel_error', '1'),
    ('CMF', '1'),
    ('T_pot', 'K'),
    ('dT_cmb', 'K'),
    ('R_cmb', 'm'),
    ('P_cmb', 'Pa'),
    ('T_cmb', 'K'),
    ('P_inner', 'Pa'),
    ('T_inner', 'K'),
)


@dataclass
class InteriorResult:
    """Output of ``interior_profiles``.

    ``line`` is the nominal ``CompatibleLine``, ``envelope`` maps 'dense' and
    'light' to the lines at the corners of the 1 sigma box (empty when no
    uncertainties are given), and ``profiles`` are sampled on the nominal line.
    """

    line: object
    envelope: dict
    profiles: list
    inputs: dict
    tables: object

    def summary(self):
        """Return the per-profile scalars as a dict of arrays (SI units)."""
        rows = [p.scalars() for p in self.profiles]
        out = {'profile': np.arange(len(rows))}
        for key, _ in SUMMARY_COLUMNS[1:]:
            if key == 'R_rel_error':
                out[key] = np.array([r['R'] / self.line.radius - 1.0 for r in rows])
            else:
                out[key] = np.array([r[key] for r in rows])
        return out

    def _header(self, title):
        mass, radius = self.inputs['mass'], self.inputs['radius']
        lo, hi = self.inputs['T_pot']
        return [
            title,
            f'paleos-interior {__version__}',
            *self.tables.provenance(),
            f'Earth units: R_earth = {R_EARTH:.5e} m, M_earth = {M_EARTH:.5e} kg',
            f'Input mass: {_fmt_uncertain(mass)} Earth masses',
            f'Input radius: {_fmt_uncertain(radius)} Earth radii',
            f'Input T_pot range: {lo:g} to {hi:g} K, dT_cmb = {self.inputs["dT_cmb"]:g} K',
            (
                'Model: Fe core under an MgSiO3 mantle, hydrostatic, adiabatic from '
                f'P = {P_SURFACE:g} Pa at T_pot'
            ),
            f'Grain size: {self.inputs["grain_size"]:g} m',
        ]

    def write(self, folder):
        """Write summary.csv, line.csv, envelope.csv, and profile_NNN.csv into folder."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        summary = self.summary()
        _write_csv(
            folder / 'summary.csv',
            self._header('Per-profile scalars, profiles evenly spaced in T_pot'),
            SUMMARY_COLUMNS,
            [summary[k] for k, _ in SUMMARY_COLUMNS],
        )
        _write_csv(
            folder / 'line.csv',
            self._header('(T_pot, CMF) pairs reproducing the nominal mass and radius'),
            LINE_COLUMNS,
            _line_arrays(self.line),
        )
        self._write_envelope(folder / 'envelope.csv')
        for k, profile in enumerate(self.profiles):
            scalars = ', '.join(
                f'{key} = {value:.6g}' for key, value in profile.scalars().items()
            )
            header = self._header(f'Radial profile {k:03d}, surface to r = {R_INNER:g} m')
            header += SOURCES
            header.append(f'Scalars (SI): {scalars}')
            _write_csv(
                folder / f'profile_{k:03d}.csv',
                header,
                COLUMNS,
                [profile.columns[key] for key, _ in COLUMNS],
            )
        return folder

    def _write_envelope(self, path):
        header = self._header(
            'Lines at the densest and least dense corners of the 1 sigma box'
        )
        corner_rows = []
        for name, line in self.envelope.items():
            M, R = line.mass / M_EARTH, line.radius / R_EARTH
            header.append(f'{name}: M = {M:.6g} Earth masses, R = {R:.6g} Earth radii')
            if not line.solvable:
                header.append(f'{name}: no Fe-MgSiO3 solution in the T_pot range')
            arrays = _line_arrays(line)
            corner_rows += [[name, *row] for row in zip(*arrays)]
        columns = (('corner', ''), *LINE_COLUMNS)
        arrays = list(zip(*corner_rows)) if corner_rows else [[] for _ in columns]
        _write_csv(path, header, columns, arrays)


def _fmt_uncertain(x):
    value, minus, plus = x
    return f'{value:.6g} (-{minus:.4g}, +{plus:.4g})'


def _line_arrays(line):
    return [line.T_pot, line.cmf, line.R_cmb, line.P_cmb, line.T_cmb]


def _write_csv(path, header, columns, arrays):
    with open(path, 'w', newline='') as f:
        f.writelines(f'# {line}\n' for line in header)
        f.write('# Units: ' + ', '.join(f'{k} [{u}]' for k, u in columns if u) + '\n')
        writer = csv.writer(f)
        writer.writerow([k for k, _ in columns])
        for row in zip(*arrays):
            writer.writerow([_fmt(v) for v in row])


def _fmt(value):
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f'{float(value):.10g}'


def interior_profiles(
    mass,
    radius,
    T_pot,
    n_profiles=20,
    dT_cmb=0.0,
    n_line=12,
    n_radial=1000,
    table_dir=None,
    grain_size=GRAIN_SIZE,
):
    """Return interior profiles compatible with a measured mass and radius.

    ``mass`` (Earth masses) and ``radius`` (Earth radii) are a value, a
    (value, sigma) pair, or (value, sigma_minus, sigma_plus). ``T_pot`` is the
    (min, max) range of 1 bar mantle potential temperatures in K. The nominal
    line is solved on ``n_line`` T_pot values, ``n_profiles`` profiles of
    ``n_radial`` points are sampled evenly in T_pot along it, and the
    uncertainties give the envelope lines. ``grain_size`` (m) sets the silicate
    viscosities and the diffusion-creep branch of solid iron. Raises
    ``NoSolutionError`` when no Fe-MgSiO3 interior reproduces the nominal mass
    and radius.
    """
    if not grain_size > 0.0:
        raise ValueError(f'grain_size must be positive, got {grain_size}')
    tables = load_tables(table_dir)
    M = _uncertain(mass, 'mass')
    R = _uncertain(radius, 'radius')
    line = compatible_line(M[0], R[0], T_pot, dT_cmb, n_line, tables)
    if any(M[1:]) or any(R[1:]):
        envelope = compatible_envelope(M, R, T_pot, dT_cmb, n_line, tables)
    else:
        envelope = {}
    profiles = line.sample(n_profiles, n_radial, grain_size)
    inputs = {
        'mass': M,
        'radius': R,
        'T_pot': tuple(sorted(T_pot)),
        'dT_cmb': dT_cmb,
        'grain_size': grain_size,
    }
    return InteriorResult(line, envelope, profiles, inputs, tables)
