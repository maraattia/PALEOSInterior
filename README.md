# PALEOSInterior

Interior profiles of rocky exoplanets compatible with a measured mass and radius. The package builds two-layer planets, an iron core under an MgSiO3 mantle, in hydrostatic equilibrium with an adiabatic temperature profile, using the PALEOS equations of state (Attia et al. 2026) through their published lookup tables. For a given mass and radius, a rocky interior is degenerate between the core mass fraction and the mantle potential temperature. The package finds the line of (potential temperature, core mass fraction) pairs that reproduce the observed radius, brackets it with the measurement uncertainties, and returns radial profiles sampled along it.

Status: in development, no usable release yet.

## Installation

```bash
pip install git+https://github.com/maraattia/PALEOSInterior.git
```

The PALEOS iron and MgSiO3 tables (about 125 MB) are downloaded from Zenodo on first use and cached locally.

## Related

- PALEOS library: https://github.com/maraattia/PALEOS
- PALEOS lookup tables: https://doi.org/10.5281/zenodo.19000315

## License

BSD 3-Clause, see `LICENSE`.
