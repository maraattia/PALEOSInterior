"""Physical constants and fixed model settings (SI units)."""

G = 6.67430e-11  # CODATA 2018, m3 kg-1 s-2
R_EARTH = 6.3781e6  # IAU 2015 Resolution B3 nominal equatorial radius, m
M_EARTH = 3.986004e14 / G  # IAU 2015 nominal GM divided by G, kg

P_SURFACE = 1e5  # Pa
R_INNER = 5e3  # profiles stop here, outside the r -> 0 coordinate singularity, m
