# PALEOSInterior

Interior profiles of rocky exoplanets compatible with a measured mass and radius. The package builds two-layer planets, an iron core under an MgSiO3 mantle, in hydrostatic equilibrium with an adiabatic temperature profile, using the PALEOS equations of state (Attia et al. 2026) through their published lookup tables. For a given mass and radius, a rocky interior is degenerate between the core mass fraction (CMF) and the mantle potential temperature, T_pot. The package finds the line of (T_pot, CMF) pairs that reproduce the observed radius, brackets it with the measurement uncertainties, and returns radial profiles sampled along it, including bulk and shear moduli, seismic wave speeds, and viscosity.

The tutorial notebook [`examples/kepler80_tutorial.ipynb`](examples/kepler80_tutorial.ipynb) goes from a single planet to the full output for the Kepler-80 planets.

## Installation

```bash
pip install git+https://github.com/maraattia/PALEOSInterior.git
```

The core package needs Python 3.10 or later, numpy, and scipy. The tutorial also needs matplotlib, pandas, and JupyterLab, which the `examples` extra installs:

```bash
pip install "paleos-interior[examples] @ git+https://github.com/maraattia/PALEOSInterior.git"
```

The PALEOS iron and MgSiO3 tables of release 1.3.0 (Zenodo, doi:10.5281/zenodo.22776069, about 125 MB) are downloaded on first use, checked against their md5 sums, and cached in `~/.cache/paleos_interior/1.3.0` (or under `$XDG_CACHE_HOME`). Set `PALEOS_INTERIOR_CACHE` to use another folder, or pass `table_dir` to read the two files from a folder of your own.

## Quick start

```python
import paleos_interior as pi

result = pi.interior_profiles(
    mass=(2.97, 0.65, 0.76),       # Earth masses: value, sigma_minus, sigma_plus
    radius=(1.330, 0.038, 0.039),  # Earth radii: same forms
    T_pot=(1000.0, 4500.0),        # K, range of mantle potential temperatures
    n_profiles=10,
)

result.line.T_pot, result.line.cmf  # the compatible line at the nominal mass and radius
result.envelope['dense']            # line at (M + sigma_M+, R - sigma_R-)
result.envelope['light']            # line at (M - sigma_M-, R + sigma_R+)
p = result.profiles[0]              # profiles evenly spaced in T_pot along the line
p.r, p.P, p.T, p.rho, p.V_S, p.eta  # SI arrays from the surface to r = 5 km
result.summary()                    # per-profile scalars as a dict of arrays
result.write('output/kepler80e')    # CSV files with a provenance header
```

Mass and radius take a single value, a `(value, sigma)` pair, or `(value, sigma_minus, sigma_plus)`. This example, Kepler-80 e with the values of MacDonald et al. (2021), takes about 30 s on a laptop once the tables are cached.

## Model

- Composition: a pure iron core under a pure MgSiO3 mantle. Neither light elements in the core nor FeO, CaO, or Al2O3 in the mantle are included.
- Structure: hydrostatic equilibrium integrated inward from the surface, $dm/dr = 4\pi r^2 \rho$ and $dP/dr = -G m \rho / r^2$ , with the radius found as the root of the mass left at the center. Profiles end at r = 5 km, outside the coordinate singularity at r = 0.
- Temperature: an adiabat, $dT/dP = \nabla_\mathrm{ad} T / P$ , from 1 bar at T_pot through the mantle and the core. The optional `dT_cmb` (K) adds a temperature jump at the core-mantle boundary (CMB) for a thermal boundary layer.
- Equations of state: the PALEOS 150 points-per-decade tables, interpolated bilinearly in log P and log T. Near the melting curve, solid and liquid are interpolated separately, each continued a few nodes past the curve, and density and the adiabatic gradient are blended linearly across 0.002 on either side of the melting temperature in log T. This keeps the structure equations continuous where an adiabat follows the curve. The heat capacities and the thermal expansion come from the point's own phase. Iron has the bcc (alpha and delta), fcc, hcp, and liquid phases, and MgSiO3 has low- and high-pressure clinoenstatite, orthoenstatite, bridgmanite, post-perovskite, and liquid. Attia et al. (2026) give the sources of each phase.
- Units: SI throughout, except masses and radii given in Earth units, which use the IAU 2015 Resolution B3 nominal values, R_earth = 6.3781e6 m (equatorial) and M_earth = GM_earth / G = 5.9722e24 kg.

### Choosing the temperature range

`T_pot` is a required `(min, max)` pair. The plausible range of the mantle potential temperature depends on the planet's age, irradiation, and tidal heating, so the package sets no default. The compatible line is solved at `n_line` (default 12) evenly spaced values across it. A hotter mantle is less dense, so at fixed mass and radius the CMF rises with T_pot, and the range chosen sets the range of CMF reported.

### Measurement uncertainties

The profiles sit on the line at the nominal mass and radius. The uncertainties enter through two more lines, at the corners of the 1 sigma box with the highest and lowest bulk density. At fixed T_pot, CMF rises with mass and falls with radius, so these corners bound the CMF of every mass and radius in the box. This holds at every point of a 5 by 5 grid across the boxes of Kepler-80 d and e at T_pot from 1000 to 4500 K, and is expected wherever the radius grows with mass at fixed composition. The uncertainties are not propagated into the profiles.

### When there is no rocky solution

If no Fe-MgSiO3 interior reproduces the mass and radius anywhere in the `T_pot` range, `compatible_line` and `interior_profiles` raise `NoSolutionError`. The message names the bounding case: the largest rocky planet of that mass (pure MgSiO3 at the hottest `T_pot`) when the radius is too large, or the smallest (pure iron) when it is too small. A radius too large for any rocky interior needs material less dense than MgSiO3, such as an atmosphere or a volatile layer, which this package does not model. When only part of the range has a solution, the line covers that part and a `UserWarning` explains the rest. An envelope corner with no solution gives an empty line, recorded in `envelope.csv`.

## Functions

| Function | Returns |
| --- | --- |
| `interior_profiles(mass, radius, T_pot, n_profiles=20, dT_cmb=0.0, n_line=12, n_radial=1000, table_dir=None, grain_size=1e-3)` | `InteriorResult` with `line`, `envelope`, `profiles`, `summary()`, and `write(folder)` |
| `compatible_line(mass, radius, T_pot, dT_cmb=0.0, n_line=12, tables=None)` | `CompatibleLine` with arrays `T_pot`, `cmf`, `R_cmb`, `P_cmb`, `T_cmb`, the solvable interval `T_range`, `cmf_at(T_pot)`, and `sample(n)` |
| `compatible_envelope(mass, radius, T_pot, dT_cmb=0.0, n_line=12, tables=None)` | dict of the `'dense'` and `'light'` corner lines |
| `integrate(mass, cmf, T_pot, dT_cmb=0.0, tables=None, n_radial=1000, grain_size=1e-3)` | one `Profile` at fixed mass, CMF, and `T_pot` |
| `load_tables(table_dir=None)` | the iron and MgSiO3 tables, to pass as `tables=` and avoid reloading |

A `Profile` exposes each column below as an array attribute (`p.rho`, `p.V_P`) and the scalars `mass`, `radius`, `cmf`, `T_pot`, `dT_cmb`, `grain_size`, `R_cmb`, `P_cmb`, `T_cmb`, `P_center`, and `T_center` in SI units. `P_center` and `T_center` are the values at r = 5 km.

## Output files

`InteriorResult.write(folder)` writes plain CSV files. Each starts with `#` comment lines giving the package version, the table release and md5 sums, the Earth units, the inputs, and the unit of every column, so any CSV reader that skips `#` lines reads them (for example, `pandas.read_csv(path, comment='#')`).

- `summary.csv`: one row per profile with `M`, `R`, `R_rel_error` (relative difference from the nominal radius), `CMF`, `T_pot`, `dT_cmb`, `R_cmb`, `P_cmb`, `T_cmb`, `P_inner`, and `T_inner` (at r = 5 km).
- `line.csv`: the nominal line, `T_pot`, `CMF`, `R_cmb`, `P_cmb`, and `T_cmb`.
- `envelope.csv`: the same columns for the two corner lines, with a `corner` column.
- `profile_NNN.csv`: one radial profile per file, from the surface to r = 5 km, with the columns below. The CMB radius appears twice, once with mantle and once with core properties. The header also lists the sources of the shear modulus and viscosity laws.

| Column | Unit | Quantity |
| --- | --- | --- |
| `r` | m | radius |
| `m` | kg | mass enclosed within `r` |
| `g` | m s⁻² | gravitational acceleration |
| `P` | Pa | pressure |
| `T` | K | temperature |
| `rho` | kg m⁻³ | density |
| `cp` | J kg⁻¹ K⁻¹ | isobaric heat capacity |
| `cv` | J kg⁻¹ K⁻¹ | isochoric heat capacity |
| `alpha` | K⁻¹ | thermal expansion coefficient |
| `nabla_ad` | 1 | adiabatic gradient, $d \ln T / d \ln P$ |
| `K_T` | Pa | isothermal bulk modulus |
| `K_S` | Pa | adiabatic bulk modulus |
| `V_phi` | m s⁻¹ | bulk sound speed, $\sqrt{K_S / \rho}$ |
| `mu` | Pa | unrelaxed shear modulus |
| `V_P` | m s⁻¹ | P-wave speed, $\sqrt{(K_S + 4 \mu_\mathrm{r} / 3) / \rho}$ |
| `V_S` | m s⁻¹ | S-wave speed, $\sqrt{\mu_\mathrm{r} / \rho}$ |
| `eta` | Pa s | viscosity |
| `layer` | | `mantle` or `core` |
| `phase` | | PALEOS phase label, `liquid` above the melting curve |

## Elastic and transport properties

The PALEOS tables give the density, the isobaric and isochoric heat capacities, the thermal expansion, and the adiabatic gradient. The bulk moduli follow from the thermodynamic identities $C_P - C_V = T \alpha^2 K_T / \rho$ and $K_S = K_T C_P / C_V$ , so they inherit any error in the tabulated heat capacities.

A point is liquid when its temperature exceeds the melting curve that PALEOS uses: Anzellini et al. (2013) for iron, and Belonoshko et al. (2005) below 2.55 GPa joined to Fei et al. (2021) above it for MgSiO3. The `phase` column is relabeled to agree with that test, so phase, shear modulus, and viscosity are always consistent.

The shear modulus `mu` is the unrelaxed (infinite-frequency) one in every phase, the parameter of a Maxwell or Andrade body. The relaxed modulus $\mu_\mathrm{r}$ in `V_P` and `V_S` equals `mu` in solids and is zero in liquids, whose Maxwell times are of order 1e-11 s or shorter, so `V_S` = 0 in melt.

- Solid silicates: the finite-strain and Debye law of Stixrude and Lithgow-Bertelloni (2024), evaluated at the PALEOS volume.
- Solid iron: the same cold curve with the thermal softening of Nadal and Le Poac (2003), parameters from the ab initio hcp-Fe moduli of Martorell et al. (2013), whose 7350 K melting point at 360 GPa is scaled with the Anzellini et al. (2013) curve.
- Liquid iron: a fit to the unrelaxed moduli of Xian et al. (2019, Table S2).
- Liquid MgSiO3: the moduli of MgSiO3 glass of Ghosh et al. (2014) at the same density.

The viscosity `eta` depends on the grain size `grain_size` (m, default 1 mm) through the solid laws.

- Silicate solids: Tackley et al. (2013), eq. 4, with the perovskite and upper-mantle rows and the upper-bound post-perovskite row of their Table 2. The reference viscosities are lowered 100 times to the Earth values they state and scale as the grain size squared. The upper-mantle row stands in for the pyroxenes, for which no dedicated law is used.
- Solid iron: the lower of Nabarro-Herring diffusion creep and dislocation creep, both from Ritterbex and Tsuchiya (2020) at equal homologous temperature. Dislocation creep does not depend on grain size and sets the value for grains above about 0.05 mm.
- Liquid iron: the free-volume model of Li et al. (2021).
- Liquid MgSiO3: a Vogel-Fulcher-Tammann law with pressure-dependent terms, fitted to the ab initio molecular dynamics viscosities of Luo and Deng (2025) and Luo et al. (2021) from 2200 to 14000 K (rms 0.096 dex).

Far below the melting temperature, the Arrhenius laws of the solids give viscosities of 1e30 Pa s and more. On tidal time scales such layers respond elastically: a layer with a viscosity of 1e25 Pa s and a shear modulus of 1e11 Pa already has a Maxwell time of 1e14 s, about 3 Myr.

## Scope

- With a pure iron core, an Earth-mass planet with CMF 0.325 at a T_pot of 1600 K has a radius of 6334 km, 0.6% below Earth's mean radius, and its core is solid throughout. Earth's core is less dense and partly liquid because light elements lower its density and its melting point.
- There are no water, ice, or atmospheric layers, so the package answers whether a planet can be rocky and, if it can, how its interior may look.
- The temperature profile is adiabatic everywhere. A conductive lid or a stagnant-lid mantle is not modeled.
- The surface pressure is 1 bar.

## Citing

Please cite the PALEOS paper, Attia, M., Lichtenberg, T., Jungová, E., & Sastre, M. 2026, A&A, 713, A225, doi:10.1051/0004-6361/202660790, and this software. `CITATION.cff` holds both, and GitHub shows them under "Cite this repository". The tables are archived on Zenodo, doi:10.5281/zenodo.22776069.

The Kepler-80 masses and radii in the tutorial and quick start are from MacDonald, M. G., Shakespeare, C. J., & Ragozzine, D. 2021, AJ, 162, 114, doi:10.3847/1538-3881/ac12d5.

## Related

- PALEOS library: https://github.com/maraattia/PALEOS
- PALEOS lookup tables: https://doi.org/10.5281/zenodo.19000315

## License

BSD 3-Clause, see `LICENSE`.
