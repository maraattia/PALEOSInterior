"""Melting curves, shear modulus, and viscosity of the PALEOS phases (SI units).

A point is molten when T exceeds the melting curve that PALEOS uses for its
phase diagram, and liquids carry G = 0. Solid silicates take the shear modulus
of Stixrude and Lithgow-Bertelloni (2024) at the PALEOS volume. Solid iron
takes the same cold curve with the homologous softening of Nadal and Le Poac
(2003), fitted to the ab initio hcp-Fe moduli of Martorell et al. (2013).
Viscosities follow the per-phase laws cited in ``viscosity``.
"""

import numpy as np

K_B = 1.380649e-23  # J K-1
R_GAS = 8.314462618  # J mol-1 K-1
EV = 1.602176634e-19  # J
AMU = 1.66053906660e-27  # kg
M_FE = 55.845 * AMU  # kg per atom

SOLID_IRON = ('solid-alpha-bcc', 'solid-delta-bcc', 'solid-gamma-fcc', 'solid-epsilon-hcp')
PYROXENES = ('solid-lpcen', 'solid-en', 'solid-hpcen')

SOURCES = (
    (
        'Melting: Anzellini et al. (2013) for Fe, Belonoshko et al. (2005) and '
        'Fei et al. (2021) for MgSiO3, as in PALEOS'
    ),
    (
        'Shear modulus: Stixrude & Lithgow-Bertelloni (2024) at the PALEOS volume, '
        'solid Fe softened after Nadal & Le Poac (2003) as fitted to Martorell et al. '
        '(2013), zero in liquids'
    ),
    (
        'Viscosity of solids: post-perovskite from Tackley et al. (2013) eq. 1 (upper '
        'bound), bridgmanite and pyroxenes from their eq. 4 lowered 100 times, Fe from '
        'Ritterbex & Tsuchiya (2020) as the lower of diffusion and dislocation creep'
    ),
    (
        'Viscosity of liquids: Fe from Li et al. (2021), MgSiO3 from a VFT fit to '
        'Luo & Deng (2025) and Luo et al. (2021)'
    ),
)


def melting_temperature(material, P):
    """Return the melting temperature (K) of 'iron' or 'mgsio3' at pressure P (Pa)."""
    P_GPa = np.asarray(P, dtype=float) / 1e9
    if material == 'iron':
        # Anzellini et al. (2013), eq. 6, both branches meet at the 98.5 GPa triple point.
        low = 1991.0 * ((P_GPa - 5.2) / 27.39 + 1.0) ** (1.0 / 2.38)
        high = 3712.0 * (np.maximum(P_GPa - 98.5, 0.0) / 161.2 + 1.0) ** (1.0 / 1.72)
        return np.where(P_GPa < 98.5, low, high)
    if material == 'mgsio3':
        # Belonoshko et al. (2005) below 2.5517 GPa, Fei et al. (2021) lower bound above.
        P_GPa = np.maximum(P_GPa, 0.0)
        low = 1831.0 * (1.0 + P_GPa / 4.6) ** 0.33
        high = 6000.0 * (P_GPa / 140.0) ** 0.26
        return np.where(P_GPa < 2.551686137257537, low, high)
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
    dE = _debye_energy(T, theta, n) - _debye_energy(np.full_like(T, 300.0), theta, n)
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
# eqs. 10 and 11). f = 0.112 is the Lindemann constant of Martorell et al. (2013); zeta
# is fitted here to their Table S1 at 360 GPa with f and T_m held at 0.112 and 7350 K.
NP_F = 0.112
NP_ZETA = 0.1048


def _iron_shear(p, P, T, rho):
    """Shear modulus (Pa) of solid iron with homologous softening up to melting."""
    That = np.minimum(T / melting_temperature('iron', P), 0.999)
    C = (6.0 * np.pi**2) ** (2.0 / 3.0) * NP_F**2 / 3.0
    J = 1.0 + np.exp(-(1.0 + 1.0 / NP_ZETA) / (1.0 + NP_ZETA / (1.0 - That)))
    G0 = _slb_cold_at_pressure(p, P)
    return (G0 * (1.0 - That) + rho * K_B * T / (C * M_FE)) / J


def shear_modulus(P, T, rho, phase):
    """Return the shear modulus (Pa) for arrays of P (Pa), T (K), rho, and phase labels."""
    P, T, rho = (np.asarray(a, dtype=float) for a in (P, T, rho))
    phase = np.asarray(phase)
    G = np.zeros_like(P)
    for label, key in _SLB_PHASE.items():
        k = phase == label
        if not k.any():
            continue
        if label in SOLID_IRON:
            G[k] = _iron_shear(_SLB[key], P[k], T[k], rho[k])
        else:
            G[k] = _slb_shear(_SLB[key], rho[k], T[k])
    return G


def _tackley_enthalpy(P, E0, V0, p_decay):
    """Activation enthalpy (J mol-1), Tackley et al. (2013) eqs. 2 and 3."""
    return E0 + P * V0 * np.exp(-P / p_decay)


def _tackley_relative(P, T, grain_size, eta_ref, E0, V0, p_decay):
    """Tackley et al. (2013) eq. 4 with T0 = 1600 K, scaled as d^2 from 1 mm."""
    H = _tackley_enthalpy(P, E0, V0, p_decay)
    return (
        eta_ref * (grain_size / 1e-3) ** 2 * np.exp(H / (R_GAS * T) - E0 / (R_GAS * 1600.0))
    )


# Post-perovskite, slow-direction (upper-bound) diffusion: Tackley et al. (2013)
# Table 1 prefactors (s m-5) at 131.6, 505.4, and 1016.1 GPa and the Table 2 fit of H.
_PPV_P = np.array([131.6e9, 505.4e9, 1016.1e9])
_PPV_ETA0 = np.array([40e30, 95e30, 180e30])
VACANCY_FRACTION = 1e-5


def _ppv_viscosity(P, T, grain_size):
    eta0 = np.exp(np.interp(P, _PPV_P, np.log(_PPV_ETA0)))
    H = _tackley_enthalpy(P, 780e3, 1.70e-6, 1100e9)
    return eta0 * grain_size**2 * K_B * T / VACANCY_FRACTION * np.exp(H / (R_GAS * T))


# hcp Fe self-diffusion at 360 GPa (Ritterbex & Tsuchiya 2020, Table 1), applied at
# the same homologous temperature at any pressure. Their melting point at 360 GPa,
# 6370 K, sets the scale, and nearest-neighbor jumps give f Z / 6 = 0.78146 x 2.
_FE_H = (10.3 + 3.21) * EV
_FE_S = 3.26
_FE_NU = 21.3e12
_FE_TM_REF = 6370.0


def _iron_diffusivity(P, T, rho):
    """Fe self-diffusion coefficient (m2 s-1) at homologous temperature T / T_m(P)."""
    T_ref = T / melting_temperature('iron', P) * _FE_TM_REF
    jump = (np.sqrt(2.0) * M_FE / rho) ** (1.0 / 3.0)
    return 0.78146 * 2.0 * jump**2 * _FE_NU * np.exp(_FE_S - _FE_H / (K_B * T_ref))


def _iron_viscosity(P, T, rho, grain_size):
    """Lower of Nabarro-Herring diffusion creep and climb-controlled dislocation creep.

    Diffusion creep is eta = d^2 k T / (alpha D Omega) with alpha = 40/3. The
    dislocation branch is the 1e17 Pa s that Ritterbex and Tsuchiya (2020) find
    for Weertman creep at 5500 K and 360 GPa, scaled as 1 / D at equal T / T_m.
    """
    D = _iron_diffusivity(P, T, rho)
    diffusion = grain_size**2 * K_B * T / (40.0 / 3.0 * D * (M_FE / rho))
    T_ref = T / melting_temperature('iron', P) * _FE_TM_REF
    dislocation = 1e17 * np.exp(_FE_H / K_B * (1.0 / T_ref - 1.0 / 5500.0))
    return np.minimum(diffusion, dislocation)


def _liquid_iron_viscosity(T, rho):
    """Free-volume fit of Li et al. (2021), eq. 12, with V0 = 6.97 A^3 and T0 = 1e4 K.

    The exponent coefficients are those that reproduce their AIMD values (Table
    I); they carry the sign of the diffusion row of Table III, v_eta = -v_D.
    """
    n = rho / M_FE
    phi = n * 6.97e-30
    v = (-0.755 + 2.764 * phi - 0.168 * phi**2) * (T / 1e4) ** -0.678
    return n ** (2.0 / 3.0) * np.sqrt(M_FE * K_B * T) * 0.313 * np.exp(v)


# ln(eta / Pa s) = a0 + a1 P + a2 P^2 + (b0 + b1 P + b2 P^2) / (T - T0), P in GPa,
# fitted to the 76 AIMD points above the PALEOS melting curve in the data of Luo &
# Deng (2025) and Luo et al. (2021), 2200 to 14000 K and 0 to 1160 GPa, rms 0.11 dex.
_MELT_VFT = (-8.00794, -0.00353864, 7.64692e-06, 5550.97, 116.863, -0.123599, 1059.74)


def _liquid_mgsio3_viscosity(P, T):
    a0, a1, a2, b0, b1, b2, T0 = _MELT_VFT
    p = P / 1e9
    return np.exp(a0 + a1 * p + a2 * p**2 + (b0 + b1 * p + b2 * p**2) / (T - T0))


def viscosity(material, P, T, rho, phase, grain_size=1e-3):
    """Return the viscosity (Pa s) for arrays of P (Pa), T (K), rho, and phase labels.

    ``grain_size`` (m) enters every solid law as d^2. Post-perovskite follows
    Tackley et al. (2013) eq. 1, slow-direction bound, with a Mg vacancy
    fraction of 1e-5. Bridgmanite and the pyroxenes follow their eq. 4 with the
    perovskite and upper-mantle rows of Table 2, lowered 100 times to the Earth
    values they state, at d = 1 mm.
    """
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
    k = phase == 'solid-ppv'
    eta[k] = _ppv_viscosity(P[k], T[k], grain_size)
    k = phase == 'solid-brg'
    eta[k] = _tackley_relative(P[k], T[k], grain_size, 3e21, 370e3, 3.65e-6, 200e9)
    k = np.isin(phase, PYROXENES)
    eta[k] = _tackley_relative(P[k], T[k], grain_size, 1e19, 300e3, 5.00e-6, np.inf)
    return eta
