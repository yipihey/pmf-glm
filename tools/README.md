# Portable Science Tools

These Python tools are independent CPU **reference models and diagnostics**, not
the nonlinear 3D PMF solver. They do not import Julia, Metal, a neighboring checkout,
private paper data, cached absolute paths, or a PDF raster. Only NumPy >= 2,
SciPy >= 1.13, and Matplotlib >= 3.8 are needed (Python >= 3.11).

## Quick Start

From the repository root:

```sh
uv sync --extra test
uv run python -m tools.reference.run_reference /tmp/pmf-reference
uv run python -m tools.reference.matched_scenario /tmp/pmf-matched
uv run python -m unittest discover -s test/python -v
# Alternatively:
uv run pytest
```

The reference command deterministically writes compact `reference.json` plus
`lmhd_p1.{png,pdf}` and `lya_hierarchy.{png,pdf}`. It compares the abrupt Schiff
TCR/FSR closure with the continuous photon P1 model at the same prescribed mode.
They are different closures, not solutions expected to agree at every epoch.
The small Ly-alpha domain is a smoke example, **not a converged paper result**.
Increase frequency extent, frequency resolution, angular order, and time
resolution independently for scientific convergence studies. No generated outputs
are checked into this extraction. JSON has no wall-clock fields; plots may have
backend/PDF metadata differences between environments.

`matched_scenario` writes `scenario.json`, compact cell-centered
`initial_fields.npz`, and `reference_modes.tsv`. Inputs explicitly state the
complex Fourier convention, geometry, comoving box size, and units. Its status is
`REFERENCE_ONLY`: no 3D solver ran. Choose `--polarization alfven` or
`magnetosonic`, `--resolution`, `--kmode`, `--box-ckpc-h`, and the redshift interval.
The NPZ contains 1D profiles broadcastable transversely, not a native restart.

The modules also support direct invocation, for example
`python tools/reference/run_reference.py /tmp/pmf-reference`. An installed package
provides `pmf-reference`, `pmf-matched-scenario`, and `pmf-analyze` commands.

## Reference Modules

Original filenames are retained under `tools/reference/`:

| Module | Retained science |
| --- | --- |
| `lmhd.py` | Schiff & Venumadhav arXiv:2506.16517v2 first-order LMHD, TCR/FSR transitions, photon P1, complex initial states, optional gravity/DM, Peebles and RECFAST-v2, local density/ionization/temperature responses |
| `lya_rt.py` | Homogeneous Fokker-Planck line, scalar A/B/C hierarchy, vector +/- hierarchy, second-order source, response-table interpolation, sparse/collocation/marching solvers |
| `sv25_paper.py` | Spectrum normalization, angular quadrature, transfer grids, density-seeded and density-aware observables, local/nonlocal perturbative recombination, visibility and mean-temperature responses |
| `build_lya_response_table.py` | Offline Ly-alpha response-table generation and interpolation checks; explicit optional external history input |
| `build_p1_meanfield_handoff.py`, `build_p1_multimode_handoff.py` | Mean-field matched-state generators, including density/temperature/chemistry and particle-unit conversions |
| `compare_matched_modes.py`, `compare_multimode_to_oracle.py` | Existing production mode/spectrum comparison contracts |

The numerical evolution algorithms in these modules are unchanged. Imports are
adapted, and the multimode component-argument adapter explicitly supplies the
newer single-mode generator's `amplitude_sign=1` (missing in the dirty source).
`matched_units.py` contains only the constants and two unit conversions
needed from the original native runner, plus positive box-input validation.
`sv25_paper.py` is the self-contained observable library, **not** the full paper
runner. The native nonlinear hierarchy validation and large campaign claims are
explicitly excluded. A scalar/vector reference BVP or a perturbative second-order
source is not a nonlinear spatial radiation-chemistry solver.

Typical small offline generation commands:

```sh
uv run python -m tools.reference.build_p1_meanfield_handoff /tmp/pmf-handoff \
  --resolution 8 --zstart 3300 --zhandoff 3200 --max-dlna 0.001
uv run python -m tools.reference.build_p1_multimode_handoff --help
uv run python -m tools.reference.build_lya_response_table /tmp/pmf-lya-table \
  --zmin 1000 --zmax 1200 --nz 3 --kmin 1000 --kmax 100000 --nk 3 \
  --xmax 20 --bins-per-width 1 --lmax 4 --workers 1
```

The last command is deliberately coarse and may fail its interpolation gate;
failure is reported by its exit status. Refine before using tables for science.
Generator defaults inherited from upstream can be expensive: specify small
resolution/ranges for smoke runs. Handoffs use raw Float32/ComplexFloat32 arrays
in Julia column-major order with `arrays.tsv` and metadata, but generate them
without Julia. They are linear, prescribed mean-field states, not certified
random-field/Batchelor pre-evolution or native 3D validation. Optional response
tables and background histories are caller-supplied files, never hidden paths.

## Mode Comparisons

The standalone `pmf-compare-nonlinear` workflow generates a complete correlated
handoff, writes a native TOML case, executes PMFGLM, and compares its actual
`metadata.toml`/`modes.tsv` to the independent photon-P1 model. It needs an
already-bootstrapped Julia backend environment and this source checkout:

```sh
uv run pmf-compare-nonlinear results/matched-metal --backend=metal \
  --resolution=32 --zend=3190 --outputs=5 --execute
```

Without `--execute` it only prepares the handoff and launch command. The default
handoff source starts at z=100000, while the native run starts at z=3200.
The generator reports the **velocity unit at the handoff**; the native case uses
that unit and retains the already physical particle-velocity convention. No
blanket velocity rescaling or fit to an evolved amplitude is performed. The
reference starts from the actual jointly imported modes to expose subsequent
evolution errors above transfer roundoff.

`--zend=500 --resolution=64` defines a longer, potentially expensive case;
repeat with smaller CFL/finer grids for convergence. `--response-table` adds
the same offline nonlocal response to the handoff, native solver, and oracle.
Without one, the case uses full local mixing chemistry with f_alpha=0.
Independent RECFAST-v2 hydrogen/temperature and Saha helium define the reference
background. Explicit helium perturbations and full-network thermal terms are
model extensions: early xHII modes and thermal perturbations are not falsely
certified against a different reduced system. A run must complete, contain
resolved evolved observables, and satisfy both amplitude and phase limits to
pass. Its `gate.json` records the input/source hashes, thresholds, and scope.

```sh
uv run python -m tools.reference.compare_reference \
  /tmp/pmf-matched/reference_modes.tsv candidate_physical_modes.tsv /tmp/pmf-compare
```

This compares `field,z,real,imag` tables with **exactly matching epochs and field
names**, preserving complex phase. Candidate velocities must already be physical
peculiar cm/s; magnetic perturbations are relative to the guide field and density
contrasts are dimensionless. No interpolation, amplitude fitting, or arbitrary
normalization is applied. A trajectory-relative floor handles zero-signal epochs.
It writes comparison JSON/TSV and PDF/PNG; a failed tolerance exits nonzero.

For the inherited native manifest contract, use
`compare_matched_modes OUTPUT RUN_DIR...` with complete `manifest.json` and
`modes.tsv` from a real controlled **linear** run. Its velocity conversion and
centered-derivative symbol are inherited intact. Full-physics runs are rejected.
`compare_multimode_to_oracle RUN HANDOFF_METADATA RESPONSE_TABLE OUTPUT` additionally
requires `summary.tsv`, `spectra.tsv`, and the explicit nonlocal response table;
it retains the upstream single-angular-bin mode contract, not general isotropic
aggregation. `tools/analysis/analyze_temporal_order.py` accepts three matched
run directories with original manifests and exact output epochs; see `--help`.

## Production TSV Analysis

One-command analysis of a completed cosmological run:

```sh
uv run pmf-analyze results/cosmology-cpu
# Or choose a separate plot directory and declare the units explicitly:
uv run pmf-analyze results/cosmology-cpu --output /tmp/pmf-analysis \
  --length-unit box --box-ckpc-h 1 --spectral-boxsize 1 --hubble 0.674
```

Without explicit TSV flags, the positional path is an input run directory.
`history.tsv`, `spectra.tsv`, and `pdfs.tsv` are discovered automatically;
the default output is `RUN_DIR/analysis`. `metadata.toml` supplies physical box,
Hubble parameter, and the numerical FFT box when available. Explicit unit flags
override inference. Input metadata is never overwritten, missing input files fail,
and noncosmological runs without a declared physical length must supply units
before their spectra can be labeled in comoving Mpc. A run still in progress is
identified by its recorded status, not presented as a completed science run.

The earlier explicit-file interface is also supported:

```sh
uv run python -m tools.analysis.analyze_production /tmp/pmf-analysis \
  --history path/to/history.tsv --spectra path/to/pkmu.tsv \
  --pdf path/to/density_pdf.tsv \
  --length-unit box --box-ckpc-h 1 --spectral-boxsize 1 --hubble 0.674
```

Each TSV is optional, but at least one is required. A local demonstration uses
the three synthetic tables in `test/python/fixtures/` with the same unit flags.
Every plot is emitted as PDF and PNG; `analysis.json` records input hashes and
units, and `isotropic_spectra.tsv` contains the weighted spectral reduction.
`spectra.{png,pdf}` summarizes up to three epochs, and
`spectra_cycleXXXXXXX_zREDSHIFT.{png,pdf}` is emitted for **every** recorded epoch.
Scalar density and summed-component magnetic power use isotropic averaging by
default. All other recorded spectral fields receive additional panels, including
gas velocity/acceleration diagnostics when present. Constant-redshift
noncosmological histories are plotted against cycle.

Schemas were inspected directly in `pmf_dm_lattice_bench.jl`:

* History: `cycle,measured,z,a,brms,divBdx_over_brms,delta_b_rms,delta_dm_rms`,
  `xHII_*`, `fH2_*`, `xHeII_*`, `xe_total_*`, transverse-field metrics,
  pressure/internal-energy statistics, and `temperature_*` (TSV, not CSV).
  Optional/disabled quantities may be NaN. Initial `measured=0` is retained.
* Spectrum: `field,cycle,measured,z,a,kbin,mubin,k,mu_lo,mu_hi,P,Nmodes`.
  Isotropic `P = sum(Nmodes * P) / sum(Nmodes)`, grouped by field, cycle, epoch,
  and radial bin. Empty bins are skipped; nonfinite populated bins and duplicate
  angular records are rejected. One-mu-bin production output is also supported.
* Density PDF: `field,cycle,measured,z,a,bin,logrho_lo,logrho_hi,rho_mid,count,pdf`.
  The PDF is volume-weighted **per log10(rho/reference density)**. Its integral is
  `sum(pdf * (logrho_hi-logrho_lo))`; finite histogram range can omit probability.

The spectrum writer uses `k = 2*pi/coordinate length` and
`P = coordinate_volume * mean(abs(normalized_Fourier_amplitude)^2)`; vector power
sums components. The header contains no length unit. **Declare it explicitly**:
`--length-unit cMpc`, `ckpc`, `ckpc/h`, or `box`. For `box`, also supply the physical
box in ckpc/h and the actual production `MHD_BOXSIZE` value. Conversion uses
`Lunit_Mpc = box_ckpc_h * 0.001 / h / spectral_boxsize`,
`k_Mpc^-1 = k_input / Lunit_Mpc`, and `P_Mpc3 = P_input * Lunit_Mpc^3`.
Thus plotted k is **comoving Mpc^-1, without h**, never physical k or h/Mpc.
`Delta2 = k^3 P/(2*pi^2)` is coordinate-unit invariant. Field amplitudes retain
production code units unless the field itself is dimensionless; the analysis does
not silently convert magnetic fields or forces to physical cgs units.

The FFT routine receives **`MHD_BOXSIZE`**, which defaults to one; it does not
receive `MHD_TERMINAL_LBOX_CKPC` directly. The latter specifies the physical
comoving box in ckpc (without h). In metadata-driven mode it is converted back to
ckpc/h before applying the formula above. Thus a 1 ckpc/h, h=0.674 box with
`MHD_BOXSIZE=1` has physical length 1.483679525 ckpc and converts raw `k` by a
factor 674 to comoving Mpc^-1.

Pressure, tension, and total Lorentz powers are plotted separately. Their powers
are not additive without the pressure/tension cross spectrum. Species fractions
are not interchanged: `xHII` is hydrogen ionization, while `xe_total` includes
additional electrons. Figures and finite-value ranges are diagnostics, not
convergence claims or automatic acceptance of production science.

## Density Frames

```sh
uv run python -m tools.analysis.render_density_frames path/to/frames /tmp/pmf-frames \
  --box-ckpc-h 1 --hubble 0.674 --log-min -2 --log-max 2
```

This reads the original `frames.tsv` schema
`cycle,z,N,slice_f32,maxproj_f32,layout` and little-endian Float32, Julia-column-major
N-by-N payloads. Each epoch produces PDF/PNG with a central density slice and a
line-of-sight **maximum**, not an integrated column density. Color limits are
fixed across frames. Without a physical box, axes use x/L and y/L; with a box,
axes use comoving kpc without h. Invalid byte counts, negative/nonfinite density,
unsupported layouts, and payload paths outside the frame directory are rejected.

## Provenance And Licenses

`tools/reference/provenance.json` lists **every copied/extracted file**, original
absolute source path, original SHA-256, delivered SHA-256, and adaptation. It
records the original dirty working-tree content rather than substituting HEAD.
It also hashes the production schema writer and chemistry main module that were
inspected but not copied. Rebuild it only when intentionally refreshing sources:

```sh
python tools/reference/record_provenance.py \
  --source /path/to/vespa-runs/tools/sv25_reference \
  --chemistry /path/to/ChemistryKernels.jl \
  --solver-schema /path/to/pmf_dm_lattice_bench.jl
```

The project-owned reference implementation is distributed here under the root
BSD-3-Clause license. Its provenance preserves the fact that the original
`vespa-runs` tree did not have a root license at extraction. External manuscripts,
the authors' repository, and third-party datasets are not relicensed by this
distribution. The RECFAST fixture's ChemistryKernels license is retained verbatim in
`test/python/fixtures/licenses/`. NumPy/SciPy/Matplotlib are separately installed,
not vendored. No PDF/raster assets, `.vendor`, campaign output, Julia source, or
native Metal dependency was copied into these tools.
