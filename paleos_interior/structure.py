"""Hydrostatic, adiabatic structure of a planet with an Fe core and an MgSiO3 mantle.

The structure equations are integrated inward from the surface,

    dm/dr = 4 pi r^2 rho,   dP/dr = -G m rho / r^2,   dT/dr = nabla_ad (T / P) dP/dr,

starting from (M, P_SURFACE, T_pot). The mantle gives way to the core where the
enclosed mass falls to CMF * M. The radius R is the root of the mass left at
R_INNER, found by bracketing and Brent's method.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .constants import M_EARTH, P_SURFACE, R_EARTH, R_INNER, G
from .profile import COLUMNS, Profile, derived_columns
from .tables import load_tables

# Radius searches use TOL_SEARCH. Profiles use TOL_PROFILE, which keeps the mass
# left at R_INNER near (4 pi / 3) rho r^3 so that P, T, and g stay smooth there.
TOL_SEARCH = 1e-9
TOL_PROFILE = 1e-11


class NoSolutionError(RuntimeError):
    """Raised when no radius reproduces the requested planet."""


def _rhs(table):
    rho_nabla = table.rho_nabla
    four_pi = 4.0 * math.pi

    def rhs(r, y):
        m, P, T = y
        if P <= 0.0 or T <= 0.0:
            return [0.0, 0.0, 0.0]
        rho, nabla = rho_nabla(P, T)
        if math.isnan(rho) or math.isnan(nabla):
            return [0.0, 0.0, 0.0]
        dP = -G * m * rho / (r * r)
        return [four_pi * r * r * rho, dP, nabla * T / P * dP]

    return rhs


def _layers(mass, cmf, tables):
    """Return (name, table, enclosed mass at the layer bottom), outermost first."""
    layers = []
    if cmf < 1.0:
        layers.append(('mantle', tables.mgsio3, mass * cmf))
    if cmf > 0.0:
        layers.append(('core', tables.iron, 0.0))
    return layers


def _integrate(R, mass, cmf, T_pot, dT_cmb, tables, tol, dense=False):
    """Integrate inward from radius R.

    Returns (residual, solutions). The residual is m(R_INNER) / M when mass is
    left over (R too small to hold M) and -r0 / R when the mass runs out at
    r0 > R_INNER (R too large). Both branches vanish at the solution.
    """
    y0 = [mass, P_SURFACE, T_pot]
    r_top = R
    solutions = []
    layers = _layers(mass, cmf, tables)
    for k, (name, table, m_bottom) in enumerate(layers):
        last = k == len(layers) - 1

        def bottom(r, y, _m=m_bottom):
            return y[0] - _m

        bottom.terminal = True
        bottom.direction = -1
        sol = solve_ivp(
            _rhs(table), (r_top, R_INNER), y0, method='RK45', rtol=tol,
            atol=tol, events=bottom, dense_output=dense or not last,
        )  # fmt: skip
        solutions.append((name, table, sol))
        if not sol.success:
            return math.nan, solutions
        hit = sol.t_events[0]
        if last:
            if hit.size:
                return -hit[0] / R, solutions
            return sol.y[0, -1] / mass, solutions
        if not hit.size:
            # The mantle alone holds more than M (1 - CMF) inside R: R is too small.
            return sol.y[0, -1] / mass, solutions
        r_top = hit[0]
        y0 = list(sol.sol(r_top))
        y0[0] = m_bottom
        y0[2] += dT_cmb
    return math.nan, solutions


def _radius_guess(mass, cmf, T_pot):
    """Rough rocky-planet radius from R ~ M^0.27 scalings, as a starting bracket."""
    f = 0.80 * cmf + 1.05 * (1.0 - cmf)
    f_hot = 1.0 + 0.05 * max(0.0, (T_pot - 500.0) / 1500.0)
    return f * f_hot * (mass / M_EARTH) ** 0.27 * R_EARTH


def _bracket(residual, R_guess, factor=1.15, max_steps=12):
    lo, hi = R_guess / factor, R_guess * factor
    f_lo, f_hi = residual(lo), residual(hi)
    for _ in range(max_steps):
        if f_hi < 0.0 < f_lo:
            return lo, hi
        if not f_lo > 0.0:
            lo /= factor
            f_lo = residual(lo)
        if not f_hi < 0.0:
            hi *= factor
            f_hi = residual(hi)
    raise NoSolutionError('no radius bracket found')


def _solve_radius(mass, cmf, T_pot, dT_cmb, tables, tol=TOL_SEARCH, guess=None, xtol=1.0):
    """Return the radius (m) of a planet of the given mass (kg), CMF, and T_pot (K).

    A ``guess`` from a looser solve narrows the starting bracket to 1e-5 of R.
    ``xtol`` (m) is the radius tolerance: 1 m moves the CMF by about 5e-7.
    """

    def residual(R):
        value, _ = _integrate(R, mass, cmf, T_pot, dT_cmb, tables, tol)
        return 1.0 if math.isnan(value) else value

    if guess is None:
        lo, hi = _bracket(residual, _radius_guess(mass, cmf, T_pot))
    else:
        lo, hi = _bracket(residual, guess, factor=1.0 + 1e-5)
    return brentq(residual, lo, hi, xtol=xtol, rtol=1e-12)


def integrate(mass, cmf, T_pot, dT_cmb=0.0, tables=None, n_radial=1000):
    """Return the converged ``Profile`` of one planet.

    ``mass`` is in Earth masses, ``T_pot`` is the 1 bar mantle potential
    temperature (K), and ``dT_cmb`` is a temperature jump added at the
    core-mantle boundary (K). Profiles run from the surface to ``R_INNER``.
    """
    if not 0.0 <= cmf <= 1.0:
        raise ValueError(f'CMF must lie in [0, 1], got {cmf}')
    tables = tables or load_tables()
    M = mass * M_EARTH
    R = _solve_radius(M, cmf, T_pot, dT_cmb, tables)
    R = _solve_radius(M, cmf, T_pot, dT_cmb, tables, TOL_PROFILE, guess=R, xtol=0.01)
    _, solutions = _integrate(R, M, cmf, T_pot, dT_cmb, tables, TOL_PROFILE, dense=True)

    tops = [R] + [sol.t_events[0][0] for _, _, sol in solutions[:-1]]
    bottoms = tops[1:] + [R_INNER]
    span = R - R_INNER
    parts = {key: [] for key, _ in COLUMNS}
    for (name, table, sol), top, bottom in zip(solutions, tops, bottoms):
        n = max(2, round(n_radial * (top - bottom) / span))
        r = np.linspace(top, bottom, n)
        m, P, T = sol.sol(r)
        props = table.evaluate(P, T)
        props['T'] = T
        cols = {'r': r, 'm': m, 'P': P, 'T': T, 'layer': np.full(n, name)}
        cols.update({k: props[k] for k in ('rho', 'cp', 'cv', 'alpha', 'nabla_ad')})
        cols['phase'] = props['phase']
        cols.update(derived_columns(r, m, props))
        for key, _ in COLUMNS:
            parts[key].append(cols[key])
    columns = {key: np.concatenate(parts[key]) for key in parts}

    profile = Profile(
        mass=M, radius=R, cmf=cmf, T_pot=T_pot, dT_cmb=dT_cmb, columns=columns
    )
    if len(solutions) == 2:
        mantle = solutions[0][2]
        profile.R_cmb = tops[1]
        profile.P_cmb, profile.T_cmb = mantle.sol(tops[1])[1:]
    return profile
