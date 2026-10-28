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
from .rheology import melt_consistent_phase, shear_modulus, viscosity
from .tables import load_tables

# Radius searches use TOL_SEARCH. Profiles use TOL_PROFILE, which keeps the mass
# left at R_INNER near (4 pi / 3) rho r^3 so that P, T, and g stay smooth there.
TOL_SEARCH = 1e-8
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


MATERIAL = {'mantle': 'mgsio3', 'core': 'iron'}


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
    left over (R too small to hold M). When the mass runs out at r0 > R_INNER
    (R too large), it is minus the mass of the empty central sphere,
    -(4 pi / 3) rho r0^3 / M. Both branches are mass excesses, so the residual
    is close to linear in R across the root.
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
                _, P0, T0 = sol.y_events[0][0]
                rho0, _ = table.rho_nabla(P0, T0)
                return -4.0 / 3.0 * math.pi * rho0 * hit[0] ** 3 / mass, solutions
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


def _secant(residual, R, xtol, max_steps=30):
    """Secant iteration from R, or None if it does not settle within max_steps."""
    f_prev = residual(R)
    # The residual changes by about -4 dR/R near the root.
    R_prev, R = R, R * (1.0 + f_prev / 4.0)
    for _ in range(max_steps):
        f = residual(R)
        if math.isnan(f) or math.isnan(f_prev) or f == f_prev:
            return None
        step = f * (R - R_prev) / (f - f_prev)
        step = min(max(step, -0.3 * R), 0.3 * R)
        if abs(step) < xtol:
            return R - step
        R_prev, f_prev, R = R, f, R - step
    return None


def _solve_radius(mass, cmf, T_pot, dT_cmb, tables, tol=TOL_SEARCH, guess=None, xtol=10.0):
    """Return the radius (m) of a planet of the given mass (kg), CMF, and T_pot (K).

    A secant iteration from ``guess`` (default: a scaling estimate) converges in
    three to five integrations. Brent's method on an expanding bracket is the
    fallback. ``xtol`` (m) is the radius tolerance: 10 m moves the CMF by about 5e-6.
    """
    start = guess or _radius_guess(mass, cmf, T_pot)

    def residual(R):
        value, _ = _integrate(R, mass, cmf, T_pot, dT_cmb, tables, tol)
        return value

    R = _secant(residual, start, xtol)
    if R is not None:
        return R

    def guarded(R):
        value = residual(R)
        return 1.0 if math.isnan(value) else value

    lo, hi = _bracket(guarded, start)
    return brentq(guarded, lo, hi, xtol=xtol, rtol=1e-12)


def integrate(mass, cmf, T_pot, dT_cmb=0.0, tables=None, n_radial=1000, grain_size=1e-3):
    """Return the converged ``Profile`` of one planet.

    ``mass`` is in Earth masses, ``T_pot`` is the 1 bar mantle potential
    temperature (K), and ``dT_cmb`` is a temperature jump added at the
    core-mantle boundary (K). ``grain_size`` (m) sets the solid viscosities.
    Profiles run from the surface to ``R_INNER``.
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
        material = MATERIAL[name]
        props = table.evaluate(P, T)
        props['T'] = T
        props['phase'] = melt_consistent_phase(material, table, P, T, props['phase'])
        props['G'] = shear_modulus(P, T, props['rho'], props['phase'])
        props['eta'] = viscosity(material, P, T, props['rho'], props['phase'], grain_size)
        cols = {'r': r, 'm': m, 'P': P, 'T': T, 'layer': np.full(n, name)}
        keys = ('rho', 'cp', 'cv', 'alpha', 'nabla_ad', 'G', 'eta', 'phase')
        cols.update({k: props[k] for k in keys})
        cols.update(derived_columns(r, m, props))
        for key, _ in COLUMNS:
            parts[key].append(cols[key])
    columns = {key: np.concatenate(parts[key]) for key in parts}

    profile = Profile(
        mass=M,
        radius=R,
        cmf=cmf,
        T_pot=T_pot,
        dT_cmb=dT_cmb,
        columns=columns,
        grain_size=grain_size,
    )
    if len(solutions) == 2:
        mantle = solutions[0][2]
        profile.R_cmb = tops[1]
        profile.P_cmb, profile.T_cmb = mantle.sol(tops[1])[1:]
    return profile
