"""The (T_pot, CMF) pairs that reproduce a planet's mass and radius.

At fixed mass, the radius falls with the core mass fraction and rises with the
potential temperature, so a measured radius picks out a line in (T_pot, CMF).
"""

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .constants import GRAIN_SIZE, M_EARTH, R_EARTH
from .structure import (
    NoSolutionError,
    OutsideTablesError,
    _check_inputs,
    _integrate,
    _solve_radius,
    integrate,
)
from .tables import Tables, load_tables

CMF_TOL = 1e-5
T_TOL = 1.0  # K

TOO_LARGE = 'larger than a pure MgSiO3 planet'
TOO_SMALL = 'smaller than a pure iron planet'
OUTSIDE = 'outside the PALEOS tables'
UNSOLVED = 'no converged structure'


class _RadiusMap:
    """Radius (m) as a function of (CMF, T_pot) at fixed mass, with memoization."""

    def __init__(self, mass, dT_cmb, tables):
        self.mass, self.dT_cmb, self.tables = mass, dT_cmb, tables
        self._cache = {}
        self._last = None

    def __call__(self, cmf, T_pot):
        key = (float(cmf), float(T_pot))
        if key not in self._cache:
            R = _solve_radius(
                self.mass, cmf, T_pot, self.dT_cmb, self.tables, guess=self._last
            )
            self._cache[key] = self._last = R
        return self._cache[key]


@dataclass
class CompatibleLine:
    """(T_pot, CMF) pairs reproducing one (mass, radius), sorted by T_pot.

    Arrays hold the solvable points only, including the ends of the solvable
    T_pot interval when it is narrower than the requested range. ``rejected``
    lists the requested T_pot values without a solution as (T_pot, reason,
    R_limit), where R_limit (m) is the radius of the pure MgSiO3 or pure iron
    planet that bounds the rocky radii at that T_pot, or NaN where no planet can
    be built because the adiabat leaves the tables or the radius search fails.
    """

    mass: float
    radius: float
    dT_cmb: float
    T_requested: tuple
    T_pot: np.ndarray
    cmf: np.ndarray
    R_cmb: np.ndarray
    P_cmb: np.ndarray
    T_cmb: np.ndarray
    rejected: list
    tables: Tables = field(repr=False)

    @property
    def solvable(self):
        return len(self.T_pot) > 0

    @property
    def T_range(self):
        """Solvable T_pot interval (K), or None."""
        return (self.T_pot[0], self.T_pot[-1]) if self.solvable else None

    def cmf_at(self, T_pot):
        """CMF on the line at T_pot, interpolated linearly between solved points."""
        return np.interp(T_pot, self.T_pot, self.cmf)

    def _cmf_exact(self, T_pot, radius_map):
        nearest = np.flatnonzero(np.isclose(self.T_pot, T_pot, rtol=0.0, atol=1e-6))
        if nearest.size:
            return float(self.cmf[nearest[0]])
        guess = float(self.cmf_at(T_pot))

        def f(c):
            return radius_map(c, T_pot) - self.radius

        for lo, hi in ((max(0.0, guess - 0.02), min(1.0, guess + 0.02)), (0.0, 1.0)):
            try:
                return brentq(f, lo, hi, xtol=CMF_TOL)
            except ValueError:
                continue
        raise NoSolutionError(
            f'no CMF reproduces the radius at T_pot = {T_pot:.0f} K, inside the sampled '
            'interval of the line'
        )

    def sample(self, n, n_radial=1000, grain_size=GRAIN_SIZE):
        """Return ``n`` profiles evenly spaced in T_pot over the solvable interval.

        ``grain_size`` (m) sets the silicate viscosities and the diffusion-creep
        branch of solid iron in every profile. A line solvable at a single T_pot
        gives one profile.
        """
        _check_inputs(self.tables, count=('n', n, 1))
        if not self.solvable:
            raise NoSolutionError('the line has no solvable point to sample')
        lo, hi = self.T_range
        if lo == hi:
            temperatures = [lo]
        else:
            temperatures = np.linspace(lo, hi, n) if n > 1 else [0.5 * (lo + hi)]
        radius_map = _RadiusMap(self.mass, self.dT_cmb, self.tables)
        profiles = []
        for T in temperatures:
            cmf = self._cmf_exact(float(T), radius_map)
            profiles.append(
                integrate(
                    self.mass / M_EARTH,
                    cmf,
                    float(T),
                    self.dT_cmb,
                    self.tables,
                    n_radial,
                    grain_size,
                )
            )
        return profiles


def _node(radius_map, R_obs, T):
    """Return (CMF, None) on the line at T, or (None, reason) if there is none."""
    try:
        if radius_map(0.0, T) < R_obs:
            return None, TOO_LARGE
        if radius_map(1.0, T) > R_obs:
            return None, TOO_SMALL
        cmf = brentq(lambda c: radius_map(c, T) - R_obs, 0.0, 1.0, xtol=CMF_TOL)
    except OutsideTablesError:
        return None, OUTSIDE
    except NoSolutionError:
        return None, UNSOLVED
    return cmf, None


def _edge(radius_map, R_obs, reason, T_a, T_b):
    """T_pot (K) between T_a and T_b where the line reaches CMF 0 or 1."""
    cmf = 0.0 if reason == TOO_LARGE else 1.0
    T = brentq(lambda t: radius_map(cmf, t) - R_obs, T_a, T_b, xtol=T_TOL)
    return T, cmf


def _cmb(radius_map, cmf, T):
    """Core-mantle boundary radius, pressure, and temperature on the mantle side."""
    if cmf <= 0.0 or cmf >= 1.0:
        return math.nan, math.nan, math.nan
    R = radius_map(cmf, T)
    _, solutions = _integrate(
        R, radius_map.mass, cmf, T, radius_map.dT_cmb, radius_map.tables, 1e-8
    )
    mantle = solutions[0][2]
    if not mantle.t_events[0].size:
        return math.nan, math.nan, math.nan
    _, P, T_cmb = mantle.y_events[0][0]
    return mantle.t_events[0][0], P, T_cmb


def _trace(mass, radius, T_pot, dT_cmb, n_line, tables):
    T_min, T_max = sorted(float(t) for t in T_pot)
    if T_min == T_max:
        _check_inputs(tables, mass, radius, T_min, ('n_line', n_line, 1))
        grid = np.array([T_min])
    else:
        _check_inputs(
            tables, mass, radius, (T_min, T_max), ('n_line for a T_pot range', n_line, 2)
        )
        grid = np.linspace(T_min, T_max, n_line)
    M, R_obs = mass * M_EARTH, radius * R_EARTH
    radius_map = _RadiusMap(M, dT_cmb, tables)
    nodes = [(T, *_node(radius_map, R_obs, T)) for T in grid]

    points = [(T, c) for T, c, _ in nodes if c is not None]
    for (T_a, _, why_a), (T_b, _, why_b) in itertools.pairwise(nodes):
        # CMF 0 and 1 edges are searched for; past an unsolvable node the line stops.
        if {why_a, why_b} & {OUTSIDE, UNSOLVED}:
            continue
        edges = {why_a, why_b} - {None} if why_a != why_b else set()
        for reason in edges:
            points.append(_edge(radius_map, R_obs, reason, T_a, T_b))
    points.sort()

    T_line = np.array([p[0] for p in points])
    cmf_line = np.array([p[1] for p in points])
    cmb = np.array([_cmb(radius_map, c, T) for T, c in points]).reshape(-1, 3)
    limit = {TOO_LARGE: 0.0, TOO_SMALL: 1.0}
    rejected = [
        (T, why, radius_map(limit[why], T) if why in limit else math.nan)
        for T, c, why in nodes
        if c is None
    ]
    return CompatibleLine(
        mass=M, radius=R_obs, dT_cmb=dT_cmb, T_requested=(T_min, T_max),
        T_pot=T_line, cmf=cmf_line, R_cmb=cmb[:, 0], P_cmb=cmb[:, 1],
        T_cmb=cmb[:, 2], rejected=rejected, tables=tables,
    )  # fmt: skip


def compatible_line(mass, radius, T_pot, dT_cmb=0.0, n_line=12, tables=None):
    """Return the ``CompatibleLine`` of a planet of given mass and radius.

    ``mass`` and ``radius`` are in Earth units, ``T_pot`` is the (min, max)
    potential temperature range in K, and ``n_line`` is the number of T_pot
    values solved across it. Raises ``NoSolutionError`` if no T_pot in the
    range has a rocky solution.
    """
    tables = tables or load_tables()
    line = _trace(mass, radius, T_pot, dT_cmb, n_line, tables)
    if line.T_requested[0] == line.T_requested[1]:
        n_line = 1
    if not line.solvable:
        raise NoSolutionError(_rejection_message(line, n_line))
    if line.rejected:
        lo, hi = line.T_range
        warnings.warn(
            f'{_rejection_message(line, n_line)} The line covers only T_pot from '
            f'{lo:.0f} to {hi:.0f} K.',
            stacklevel=2,
        )
    return line


def _rejection_message(line, n_line):
    """Explain the T_pot values without a rocky solution in physical terms."""
    lo, hi = line.T_requested
    head = (
        f'For M = {line.mass / M_EARTH:.4g} Earth masses and R = '
        f'{line.radius / R_EARTH:.4f} Earth radii, {len(line.rejected)} of the '
        f'{n_line} T_pot values solved from {lo:.0f} to {hi:.0f} K have no Fe-MgSiO3 '
        'solution.'
    )
    parts = [head]
    too_large = [x for x in line.rejected if x[1] == TOO_LARGE]
    too_small = [x for x in line.rejected if x[1] == TOO_SMALL]
    if too_large:
        T, _, R_lim = max(too_large, key=lambda x: x[2])
        deficit = 1.0 - R_lim / line.radius
        parts.append(
            f'At {_span([x[0] for x in too_large])} even a pure MgSiO3 planet is too small: '
            f'at T_pot = {T:.0f} K it has R = {R_lim / R_EARTH:.4f} Earth radii, '
            f'{100 * deficit:.1f}% below the measured radius. There, the measured radius '
            'needs material less dense than MgSiO3, such as an atmosphere or a volatile '
            'layer, above a smaller Fe-MgSiO3 interior.'
        )
    if too_small:
        T, _, R_lim = min(too_small, key=lambda x: x[2])
        excess = R_lim / line.radius - 1.0
        parts.append(
            f'At {_span([x[0] for x in too_small])} even a pure iron planet is too large: '
            f'at T_pot = {T:.0f} K it has R = {R_lim / R_EARTH:.4f} Earth radii, '
            f'{100 * excess:.1f}% above the measured radius, so there the planet is denser '
            'than pure iron allows.'
        )
    outside = [x[0] for x in line.rejected if x[1] == OUTSIDE]
    if outside:
        parts.append(
            f'At {_span(outside)} the adiabat enters a region where the PALEOS tables '
            'have no data, so no planet can be built there.'
        )
    unsolved = [x[0] for x in line.rejected if x[1] == UNSOLVED]
    if unsolved:
        parts.append(f'At {_span(unsolved)} the radius search did not converge.')
    return ' '.join(parts)


def _span(temperatures):
    lo, hi = min(temperatures), max(temperatures)
    return f'T_pot = {lo:.0f} K' if lo == hi else f'T_pot from {lo:.0f} to {hi:.0f} K'


def _uncertain(x, name):
    """Return (value, sigma_minus, sigma_plus) from a number or a 2- or 3-tuple."""
    if np.isscalar(x):
        return float(x), 0.0, 0.0
    x = tuple(float(v) for v in x)
    if len(x) == 2:
        return x[0], x[1], x[1]
    if len(x) == 3:
        return x
    raise ValueError(f'{name} must be a value, (value, sigma), or (value, minus, plus)')


def compatible_envelope(mass, radius, T_pot, dT_cmb=0.0, n_line=12, tables=None):
    """Return the lines at the densest and least dense corners of the 1 sigma box.

    'dense' is (M + sigma_M, R - sigma_R) and 'light' is (M - sigma_M, R + sigma_R).
    A corner without any solution gives a line with no solvable point.
    """
    tables = tables or load_tables()
    M, M_minus, M_plus = _uncertain(mass, 'mass')
    R, R_minus, R_plus = _uncertain(radius, 'radius')
    corners = {'dense': (M + M_plus, R - R_minus), 'light': (M - M_minus, R + R_plus)}
    for key, (m, r) in corners.items():
        if not (m > 0.0 and r > 0.0):
            raise ValueError(
                f'the {key} corner has M = {m:g} and R = {r:g}, not both positive'
            )
    return {
        key: _trace(m, r, T_pot, dT_cmb, n_line, tables) for key, (m, r) in corners.items()
    }
