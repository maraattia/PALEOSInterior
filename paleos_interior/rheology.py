"""Melting curves, shear modulus, and viscosity of the PALEOS phases (SI units).

A point is molten when T exceeds the melting curve that PALEOS uses for its
phase diagram, and liquids carry a zero shear modulus. Solid silicates take the
shear modulus of Stixrude and Lithgow-Bertelloni (2024) at the PALEOS volume.
Solid iron takes the same cold curve with the softening of Nadal and Le Poac
(2003), whose parameters come from the ab initio hcp-Fe moduli of Martorell et
al. (2013). Its melting temperature is their 7350 K at 360 GPa, carried to other
pressures in proportion to the Anzellini et al. (2013) melting curve. The viscosity laws and their sources are listed in ``viscosity``.
"""

import numpy as np

from .constants import AMU, EV, GRAIN_SIZE, K_B, R_GAS

M_FE = 55.845 * AMU  # kg per atom

SOLID_IRON = ('solid-alpha-bcc', 'solid-delta-bcc', 'solid-gamma-fcc', 'solid-epsilon-hcp')
PYROXENES = ('solid-lpcen', 'solid-en', 'solid-hpcen')

SOURCES = (
    (
        'Melting: Anzellini et al. (2013) for Fe, Belonoshko et al. (2005) and '
        'Fei et al. (2021) for MgSiO3, as in PALEOS'
    ),
    (
        'Shear modulus: Stixrude & Lithgow-Bertelloni (2024) at the PALEOS volume. Solid '
        'Fe uses their cold curve with the Nadal & Le Poac (2003) softening, parameters '
        'from Martorell et al. (2013), whose 7350 K melting point at 360 GPa scales with '
        'the Anzellini et al. (2013) curve. Zero in liquids'
    ),
    (
        'Viscosity of silicate solids: Tackley et al. (2013) eq. 4 with the upper-bound '
        'post-perovskite, perovskite, and upper-mantle rows of their Table 2, reference '
        'viscosity lowered 100 times, at 1600 K and 1 mm grains, scaled as grain size squared'
    ),
    (
        'Viscosity of solid Fe: lower of Nabarro-Herring diffusion creep and dislocation '
        'creep from Ritterbex & Tsuchiya (2020) at equal T/T_m. Dislocation creep does not '
        'depend on grain size and sets the value for grains above about 0.3 mm'
    ),
    (
        'Viscosity of liquids: Fe from the Li et al. (2021) free-volume model with the '
        'exponent sign that reproduces their AIMD values, MgSiO3 from a fit to Luo & Deng '
        '(2025) and Luo et al. (2021)'
    ),
)

# Anzellini et al. (2013) Simon-Glatzel fit for Fe, two branches meeting at the
# gamma-epsilon-liquid triple point: (P_ref GPa, T_ref K, a GPa, c) per branch.
_FE_MELT_LOW = (5.2, 1991.0, 27.39, 2.38)
_FE_MELT_HIGH = (98.5, 3712.0, 161.2, 1.72)
# MgSiO3: Belonoshko et al. (2005) below P0, Fei et al. (2021) lower bound above.
_P0_MELT_MGSIO3 = 2.551686137257537  # GPa, where the two branches meet


def melting_temperature(material, P):
    """Return the melting temperature (K) of 'iron' or 'mgsio3' at pressure P (Pa)."""
    P_GPa = np.asarray(P, dtype=float) / 1e9
    if material == 'iron':
        p0, t0, a, c = _FE_MELT_LOW
        low = t0 * ((P_GPa - p0) / a + 1.0) ** (1.0 / c)
        p0, t0, a, c = _FE_MELT_HIGH
        high = t0 * (np.maximum(P_GPa - p0, 0.0) / a + 1.0) ** (1.0 / c)
        return np.where(P_GPa < _FE_MELT_HIGH[0], low, high)
    if material == 'mgsio3':
        P_GPa = np.maximum(P_GPa, 0.0)
        low = 1831.0 * (1.0 + P_GPa / 4.6) ** 0.33
        high = 6000.0 * (P_GPa / 140.0) ** 0.26
        return np.where(P_GPa < _P0_MELT_MGSIO3, low, high)
    raise ValueError(f'unknown material {material!r}')


def melt_consistent_phase(material, table, P, T, phase):
    """Return phase labels that agree with the melting curve.

    Table labels come from the nearest grid node, so close to the melting curve
    they can disagree with T. Points above the curve become 'liquid', and points
    below it labeled 'liquid' take the nearest solid label at lower T.
    """
    P, T = np.asarray(P, dtype=float), np.asarray(T, dtype=float)
    molten = T > melting_temperature(material, P)
    out = np.where(molten, 'liquid', np.asarray(phase)).astype('<U32')
    k = ~molten & ((out == 'liquid') | (out == ''))
    if k.any():
        out[k] = table.solid_phase(P[k], T[k])
    return out


# Stixrude and Lithgow-Bertelloni (2024) parameters, copied unmodified from the
# HeFESTo set as distributed in BurnMan (burnman/minerals/SLB_2024.py):
# V0 (m3 mol-1), K0 (Pa), K0', theta0 (K), gamma0, q0, G0 (Pa), G0', eta_S0, n, molar mass (kg mol-1).
_SLB = {
    'fea': (7.093e-06, 163.42114e9, 6.01732, 398.00486, 1.66729, 0.90658, 81.47e9, 1.9223, 5.96207, 1, 0.05584515),
    'feg': (6.9883e-06, 165.0e9, 5.97, 285.19315, 1.84924, 1.02176, 91.0e9, 2.33869, 6.50014, 1, 0.05584515),
    'fee': (6.7651e-06, 165.0e9, 4.97, 379.74281, 2.13092, 1.02253, 91.0e9, 2.24804, 5.20603, 1, 0.05584515),
    'cen': (6.25e-05, 113.7599e9, 4.8061, 805.59286, 1.00921, 0.60142, 76.81031e9, 1.71384, 1.42387, 10, 0.2007774),
    'en': (6.2676e-05, 107.07681e9, 7.02751, 812.21227, 0.78477, 3.43847, 76.8e9, 1.54596, 2.5045, 10, 0.2007774),
    'mgc2': (6.076e-05, 116.02556e9, 6.23685, 824.44051, 1.12478, 0.20409, 87.92717e9, 1.84119, 2.14193, 10, 0.2007774),
    'mgpv': (2.4445e-05, 250.56535e9, 4.13438, 892.95164, 1.54466, 0.83352, 172.9e9, 1.73254, 1.65233, 5, 0.100389),
    'mppv': (2.41108e-05, 247.74e9, 4.0, 931.02549, 1.64015, 1.76454, 166.64493e9, 1.98845, 1.43815, 5, 0.100389),
}  # fmt: skip
_SLB_PHASE = {
    'solid-alpha-bcc': 'fea',
    'solid-delta-bcc': 'fea',
    'solid-gamma-fcc': 'feg',
    'solid-epsilon-hcp': 'fee',
    'solid-lpcen': 'cen',
    'solid-en': 'en',
    'solid-hpcen': 'mgc2',
    'solid-brg': 'mgpv',
    'solid-ppv': 'mppv',
}
_T_SLB_REF = 300.0  # K, reference temperature of the SLB thermal terms

_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def _debye_energy(T, theta, n):
    """Debye thermal energy 9 n R T (T/theta)^3 int_0^{theta/T} t^3 / (e^t - 1) dt (J mol-1)."""
    x = theta / T
    t = 0.5 * x[..., None] * (_GL_X + 1.0)
    integral = 0.5 * x * np.sum(_GL_W * t**3 / np.expm1(t), axis=-1)
    return 9.0 * n * R_GAS * T * integral / x**3


def _slb_cold(p, x):
    """Third-order finite-strain shear modulus (Pa) at compression x = V0 / V."""
    _, K0, Kp, _, _, _, G0, Gp, _, _, _ = p
    f = 0.5 * (x ** (2.0 / 3.0) - 1.0)
    a = 3.0 * K0 * Gp - 5.0 * G0
    b = 6.0 * K0 * Gp - 24.0 * K0 - 14.0 * G0 + 4.5 * K0 * Kp
    return (1.0 + 2.0 * f) ** 2.5 * (G0 + a * f + b * f * f)


def _slb_shear(p, rho, T):
    """SLB shear modulus (Pa) at the volume set by density rho and temperature T.

    Stixrude and Lithgow-Bertelloni (2005), eqs. 33, 41, 46, and 47, with the
    eta_S sign as implemented in BurnMan (eos/slb.py).
    """
    V0, _, _, theta0, g0, q0, _, _, eta0, n, molar_mass = p
    V = molar_mass / rho
    x = V0 / V
    f = 0.5 * (x ** (2.0 / 3.0) - 1.0)
    a1 = 6.0 * g0
    a2 = -12.0 * g0 + 36.0 * g0**2 - 18.0 * q0 * g0
    nu2 = 1.0 + a1 * f + 0.5 * a2 * f * f
    gamma = (2.0 * f + 1.0) * (a1 + a2 * f) / (6.0 * nu2)
    eta_s = -gamma - 0.5 / nu2 * (2.0 * f + 1.0) ** 2 * (-2.0 * g0 - 2.0 * eta0)
    theta = theta0 * np.sqrt(nu2)
    T_ref = np.full_like(T, _T_SLB_REF)
    dE = _debye_energy(T, theta, n) - _debye_energy(T_ref, theta, n)
    return _slb_cold(p, x) - eta_s * dE / V


def _slb_cold_at_pressure(p, P):
    """Shear modulus (Pa) on the 300 K isotherm of the SLB equation of state at P (Pa)."""
    _, K0, Kp, *_ = p
    lo, hi = np.zeros_like(P), np.full_like(P, 2.0)
    for _ in range(60):
        f = 0.5 * (lo + hi)
        P_f = 3.0 * K0 * f * (1.0 + 2.0 * f) ** 2.5 * (1.0 + 1.5 * (Kp - 4.0) * f)
        lo, hi = np.where(P_f < P, f, lo), np.where(P_f < P, hi, f)
    x = (1.0 + 2.0 * 0.5 * (lo + hi)) ** 1.5
    return _slb_cold(p, x)


# Nadal and Le Poac (2003) model as written by Banerjee (2005, arXiv:cond-mat/0512466,
# eqs. 10 and 11). f = 0.112 is the Lindemann constant of Martorell et al. (2013), and
# zeta fits their Table S1 at 360 GPa with f and T_m held at 0.112 and 7350 K.
NP_F = 0.112
NP_ZETA = 0.1048
_T_HAT_MAX = 0.999  # keeps J finite for any T up to the model melting temperature
# Their 7350 K is where the simulated crystal fails, above the thermodynamic melting
# point. The ratio to the Anzellini et al. (2013) curve at 360 GPa, 1.13, is held at all P.
_NP_TM_SCALE = 7350.0 / float(melting_temperature('iron', 360e9))


def _iron_shear(p, P, T, rho):
    """Shear modulus (Pa) of solid iron, softened toward the model melting temperature."""
    That = np.minimum(T / (_NP_TM_SCALE * melting_temperature('iron', P)), _T_HAT_MAX)
    C = (6.0 * np.pi**2) ** (2.0 / 3.0) * NP_F**2 / 3.0
    J = 1.0 + np.exp(-(1.0 + 1.0 / NP_ZETA) / (1.0 + NP_ZETA / (1.0 - That)))
    G0 = _slb_cold_at_pressure(p, P)
    return (G0 * (1.0 - That) + rho * K_B * T / (C * M_FE)) / J


def shear_modulus(P, T, rho, phase):
    """Return the shear modulus (Pa) for arrays of P (Pa), T (K), rho, and phase labels.

    Liquids get zero and points without a phase label (no table data) NaN.
    Raises ``ValueError`` for a label that is neither a known solid nor 'liquid'.
    """
    P, T, rho = (np.asarray(a, dtype=float) for a in (P, T, rho))
    phase = np.asarray(phase)
    unknown = set(np.unique(phase)) - set(_SLB_PHASE) - {'liquid', ''}
    if unknown:
        raise ValueError(f'no shear modulus for phase labels {sorted(map(str, unknown))}')
    mu = np.full_like(P, np.nan)
    mu[phase == 'liquid'] = 0.0
    for label, key in _SLB_PHASE.items():
        k = phase == label
        if not k.any():
            continue
        if label in SOLID_IRON:
            mu[k] = _iron_shear(_SLB[key], P[k], T[k], rho[k])
        else:
            mu[k] = _slb_shear(_SLB[key], rho[k], T[k])
    return mu


# Tackley et al. (2013) Table 2 rows (eta_ref Pa s, E0 J mol-1, V0 m3 mol-1, p_decay Pa),
# with eta_ref lowered 100 times to the Earth values they state, at T0 = 1600 K and
# 1 mm grains. Post-perovskite takes the upper-bound (slow diffusion) row.
_TACKLEY = {
    'upper mantle': (1e19, 300e3, 5.00e-6, np.inf),
    'perovskite': (3e21, 370e3, 3.65e-6, 200e9),
    'post-perovskite': (1.05e32, 780e3, 1.70e-6, 1100e9),
}
_TACKLEY_T0 = 1600.0  # K
_GRAIN_REF = 1e-3  # m


def _tackley_viscosity(row, P, T, grain_size):
    """Tackley et al. (2013) eq. 4 with H(P) from their eqs. 2 and 3, scaled as d^2."""
    eta_ref, E0, V0, p_decay = _TACKLEY[row]
    H = E0 + P * V0 * np.exp(-P / p_decay)
    arrhenius = np.exp(H / (R_GAS * T) - E0 / (R_GAS * _TACKLEY_T0))
    return eta_ref * (grain_size / _GRAIN_REF) ** 2 * arrhenius


# hcp Fe self-diffusion at 360 GPa (Ritterbex & Tsuchiya 2020, Table 1), applied at
# the same homologous temperature at any pressure. Their melting point at 360 GPa,
# 6370 K, sets the scale, and nearest-neighbor jumps give f Z / 6 = 0.78146 x 2.
_FE_H = (10.3 + 3.21) * EV
_FE_S = 3.26
_FE_NU = 21.3e12
_FE_JUMP_FACTOR = 0.78146 * 2.0
_FE_TM_REF = 6370.0
_NH_ALPHA = 40.0 / 3.0  # Nabarro-Herring geometric factor with grain-boundary sliding
# Weertman creep viscosity of hcp Fe at 360 GPa and 5500 K (Ritterbex & Tsuchiya 2020).
_FE_DISLOCATION = (1e17, 5500.0)


def _iron_diffusivity(T_ref, rho):
    """Fe self-diffusion coefficient (m2 s-1) at the 360 GPa equivalent temperature T_ref."""
    jump = (np.sqrt(2.0) * M_FE / rho) ** (1.0 / 3.0)
    return _FE_JUMP_FACTOR * jump**2 * _FE_NU * np.exp(_FE_S - _FE_H / (K_B * T_ref))


def _iron_viscosity(P, T, rho, grain_size):
    """Lower of Nabarro-Herring diffusion creep and climb-controlled dislocation creep.

    Diffusion creep is eta = d^2 k T / (alpha D Omega) with alpha = 40/3. The
    dislocation branch is the Weertman-creep viscosity of Ritterbex and Tsuchiya
    (2020) at 5500 K and 360 GPa, scaled as 1 / D at equal T / T_m. It does not
    depend on grain size and is the lower branch for grains above about 0.3 mm.
    """
    T_ref = T / melting_temperature('iron', P) * _FE_TM_REF
    D = _iron_diffusivity(T_ref, rho)
    diffusion = grain_size**2 * K_B * T / (_NH_ALPHA * D * (M_FE / rho))
    eta_dis, T_dis = _FE_DISLOCATION
    dislocation = eta_dis * np.exp(_FE_H / K_B * (1.0 / T_ref - 1.0 / T_dis))
    return np.minimum(diffusion, dislocation)


# Li et al. (2021) free-volume model, eq. 12: (C_eta, a0, a1, a2, alpha, V0 m3, T0 K).
# The a coefficients are those that reproduce their AIMD values (Table I). They carry
# the sign of the diffusion row of Table III, following their v_eta = -v_D.
_LI21 = (0.313, -0.755, 2.764, -0.168, -0.678, 6.97e-30, 1e4)


def _liquid_iron_viscosity(T, rho):
    """Liquid Fe viscosity (Pa s) from the free-volume model of Li et al. (2021)."""
    C, a0, a1, a2, alpha, V0, T0 = _LI21
    n = rho / M_FE
    phi = n * V0
    v = (a0 + a1 * phi + a2 * phi**2) * (T / T0) ** alpha
    return n ** (2.0 / 3.0) * np.sqrt(M_FE * K_B * T) * C * np.exp(v)


# ln(eta / Pa s) = a0 + a1 P + b0 (1 + P / P_b)^beta / (T - T0), P in GPa, fitted to the
# 76 AIMD points above the PALEOS melting curve in Luo & Deng (2025) and Luo et al. (2021),
# 2200 to 14000 K and 0 to 1160 GPa, rms 0.096 dex. The numerator stays positive at any P.
_MELT_VFT = (-8.21241, 0.000273733, 5892.01, 15.8716, 0.498385, 1018.24)


def _liquid_mgsio3_viscosity(P, T):
    a0, a1, b0, P_b, beta, T0 = _MELT_VFT
    p = P / 1e9
    return np.exp(a0 + a1 * p + b0 * (1.0 + p / P_b) ** beta / (T - T0))


def viscosity(material, P, T, rho, phase, grain_size=GRAIN_SIZE):
    """Return the viscosity (Pa s) for arrays of P (Pa), T (K), rho, and phase labels.

    Silicate solids follow Tackley et al. (2013) eq. 4 with the upper-bound
    post-perovskite, perovskite, and upper-mantle rows of their Table 2, lowered
    100 times to the Earth values they state, at 1600 K and d = 1 mm, and scale as
    d^2 with ``grain_size`` d (m). Solid Fe takes the lower of diffusion and
    dislocation creep after Ritterbex and Tsuchiya (2020), and only its diffusion
    branch depends on d. Liquid Fe follows Li et al. (2021) and liquid MgSiO3 a
    fit to Luo and Deng (2025) and Luo et al. (2021). Points without a phase
    label are NaN.
    """
    if material not in ('iron', 'mgsio3'):
        raise ValueError(f'unknown material {material!r}')
    P, T, rho = (np.asarray(a, dtype=float) for a in (P, T, rho))
    phase = np.asarray(phase)
    eta = np.full_like(P, np.nan)
    liquid = phase == 'liquid'
    if material == 'iron':
        eta[liquid] = _liquid_iron_viscosity(T[liquid], rho[liquid])
        k = np.isin(phase, SOLID_IRON)
        eta[k] = _iron_viscosity(P[k], T[k], rho[k], grain_size)
        return eta
    eta[liquid] = _liquid_mgsio3_viscosity(P[liquid], T[liquid])
    silicates = (
        (('solid-ppv',), 'post-perovskite'),
        (('solid-brg',), 'perovskite'),
        (PYROXENES, 'upper mantle'),
    )
    for labels, row in silicates:
        k = np.isin(phase, labels)
        eta[k] = _tackley_viscosity(row, P[k], T[k], grain_size)
    return eta
