# Agent Guide

These instructions apply throughout this repository. Prioritize physical
correctness, reproducibility, and measured performance, in that order. A run
that finishes without NaNs is not, by itself, a validated physical calculation.

## Orient Before Changing Anything

- Read [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), and the
  relevant [physics](docs/physics.qmd), [notation](docs/notation.qmd), and
  [validation](docs/validation.qmd) sections. Inspect the current worktree and
  preserve unrelated edits, failed runs, and negative results.
- This is standalone software. Edit the owning bundled package under `lib/`,
  orchestration under `src/`, CLI wrappers under `scripts/`, and independent
  references/analysis under `tools/`. Do not fix an unrelated sibling checkout
  or introduce a dependency on Vespa, private data, or machine-specific paths.
- Use Julia 1.12+ and the correct active project. Julia `[sources]` entries
  resolve from the active root; do not assume a `Pkg.test()` sandbox inherits
  them. Keep the root and optional backend projects consistent when changing
  dependencies. Preserve bundled licenses and extraction provenance.

## Run, Inspect, Analyze

Run commands from the repository root. Use a new output directory for every
experiment; replace the example path below if it already exists.

```sh
julia --project=. -t auto scripts/bootstrap.jl --backend=cpu
uv sync --extra test
julia --project=. -t auto scripts/run_pmf.jl examples/cosmology-smoke.toml \
  --backend=cpu --output=results/agent-smoke-cpu --dry-run
julia --project=. -t auto scripts/run_pmf.jl examples/cosmology-smoke.toml \
  --backend=cpu --output=results/agent-smoke-cpu
uv run pmf-analyze results/agent-smoke-cpu
```

This is a small full-physics installation check, not a recombination campaign.
Inspect `metadata.toml`, `resolved-environment.toml`, `run.log`, and the actual
final redshift before interpreting its `analysis/` figures. `COMPLETED` means
the target was reached; `STOPPED_AT_LIMIT`, `INCOMPLETE`, and `FAILED` do not.
Exit code zero or a final spectrum alone is insufficient. A plot is not an
automatic physics gate. See [output and restart contracts](docs/outputs.qmd).

For Metal, bootstrap its environment, verify real hardware, then run there:

```sh
julia --project=. -t 4 scripts/bootstrap.jl --backend=metal
julia --project=environments/metal -t 4 scripts/test_gpu.jl --backend=metal
julia --project=environments/metal -t 4 scripts/run_pmf.jl \
  examples/cosmology-smoke.toml --backend=metal --output=results/agent-smoke-metal
```

For CUDA, replace `metal` with `cuda` in the backend, environment, and output
path. Never silently run on CPU if the requested GPU is unavailable. Metal
uses Float32 and MPS real FFTs; CUDA uses cuFFT. Do not select the portable KA
FFT fallback as an unnoticed performance workaround. Backend support in source
is not hardware validation; label unavailable hardware `NOT_RUN`.

Use parallel Julia execution, including bootstrap/precompilation, rather than
unintentionally serial builds. FFTW has a separate thread count:
`PMF_FFTW_THREADS=4` is the starting choice on this Mac, not a universal optimum.
Inspect [configuration precedence](docs/configuration.qmd): `[environment]`
overrides named TOML options, and unrelated shell `MHD_*` options are stripped
from the child. Review the resolved configuration rather than guessing defaults.

## Scientific Guardrails

- Keep full MHD with stiff drag, evolved photon moments, full mixing chemistry
  with expansion and the declared helium treatment as the cosmological research
  baseline. Terminal/hybrid modes and ionized/analytic shortcuts are explicit
  comparison models, not automatic speed optimizations. Do not change source
  cadence, floors, fallback fluxes, or chemistry accuracy without validation.
- `magnetic.b0_ng` is **comoving nG**: 1 pG = 0.001 nG.
  `cosmology.box_ckpc_h` is **comoving kpc/h**, not physical kpc or ckpc without h.
  Physical B scales as `(1+z)^2` at fixed comoving B. Do not invent pre-start
  evolution by relabeling a field's normalization epoch. A magnetic-only import
  is not a correlated gas/photon/particle/chemistry handoff or a checkpoint.
- For grid convergence, fix the physical box, low-mode complex coefficients,
  normalization band, seed, cutoff/window, curl symbol/reference grid, and mode
  locks. The same seed alone is insufficient. Scale smoothing cells with N to
  keep the physical smoothing length fixed. Verify the initial overlap modes.
- For UV sensitivity, add modes without renormalizing the shared low-k field;
  use the documented reference-band normalization. A Nyquist-extended spectrum
  changes the physical problem and remains UV sensitivity unless a physical
  damping cutoff is specified. Maxwell stresses reach about twice the magnetic
  wavenumber: resolving B alone does not prove force convergence.
- Refine timestep and grid separately, and compare actual common redshifts.
  Define tolerances and resolvable signal floors before measuring discrepancies.
  Preserve phase information; do not fit away evolved amplitude errors. Check
  CPU Float64 where supported and CPU Float32 for GPU parity. See
  [matched ICs](docs/initial-conditions.qmd) and the [science matrix](docs/validation.qmd).

## Analyze the Actual Outputs

- Retain raw tables alongside figures. Compare B, gas/DM density, photon
  density/velocity/slip, xHII, total electron fraction, temperature, PDFs,
  divergence, and normalized energy residuals as applicable to the experiment.
  Disabled channels can be NaN; do not replace them with zero. Distinguish xHII
  from helium-inclusive electron abundance, and histogram tails from lost support.
- Use isotropic B and density spectra by default. Angular diagnostics are
  opt-in for an explicit question, not an automatic campaign expense. The
  analysis tool weights angular reductions by `Nmodes` and skips empty bins.
- The spectral coordinate box is `MHD_BOXSIZE` (default 1), not automatically
  the physical TOML box. Use metadata-aware `pmf-analyze` or explicit documented
  unit flags. Label k, P, and dimensionless power correctly; magnetic/force
  amplitudes are not silently converted to cgs. Batchelor shell energy
  `E_B(k) ~ k^4` is not the same slope as per-mode `P_B(k)`.
- Compare magnetic-pressure, tension, and total Lorentz-force spectra with
  their cancellation: `P_L = P_P + P_T + 2 P_PT`. Auto-powers alone are not
  additive. Density projections must distinguish a maximum from a column
  integral; keep movie color limits fixed. Details: [analysis tools](tools/README.md).

## Verification and Evidence

For relevant changes, run focused regressions and the appropriate checks:

```sh
julia --project=. -t auto test/runtests.jl
uv run python -m pytest
uv run python scripts/check_portability.py
```

Run the affected GPU gate on real hardware for device/kernel changes. For
coupled evolution or checkpoint changes also run `scripts/validate_production.jl`
with the appropriate `--backend` in the matching project. That command deliberately
refreshes tracked `docs/data/production-*.toml`; inspect its diff. Similarly,
`scripts/validate.jl` writes tracked evidence by default; pass
`--output=results/my-validation` for scratch validation instead. Never fabricate,
relabel, or hand-edit a numerical PASS, source hash, or measured timing. Old
evidence belongs to its recorded source/backend/scope; missing is not passing.

The Python references are independent controlled models, not the nonlinear 3D
solver. `pmf-matched-scenario` is `REFERENCE_ONLY`; the real native comparison
is `pmf-compare-nonlinear ... --execute`, requiring a bootstrapped backend.
See [its workflow and gates](docs/validation.qmd#end-to-end-native-mhd-comparison).
A short matched-mode PASS does not certify evolution through recombination.
P1 photon moments are not a full angular/polarization hierarchy or nonlinear
nonlocal Ly-alpha transport; Gaussian chemistry smoothing is not line transport.
Do not claim reproduction of every Schiff observable from selected linear tests.

## Performance, Campaigns, and Recovery

- Start small, dry-run suites, and estimate memory/wall time before increasing
  resolution or queueing a campaign. Use `MHD_MEMPROBE=1` to measure live/peak
  memory, including FFT workspace, particles, diagnostics, and checkpoint staging.
  Do not extrapolate a safe grid from nominal VRAM/unified memory alone.
- Keep evolving fields and reductions on device, reuse compatible scratch,
  and copy compact diagnostics. Shared buffers require a proven nonoverlapping
  lifetime; FFT plans/workspaces must respect device, shape, layout, and type.
- Synchronize GPU timings. Separate compilation/startup, warm MHD-only Mcells/s,
  coupled phase costs, diagnostic/I/O cost, and end-to-end progress. Radiation
  and chemistry subphase timings are nested: do not add them twice. Record
  hardware, N, threads, integrator, precision, CFL/source controls, and cadence.
  Keep `MHD_WARMUP=0` for physical trajectories; use profiling skip controls.
  [Benchmark definitions and commands](docs/performance.qmd) are authoritative.
- Run campaign cases serially on one GPU unless intentionally benchmarking
  concurrency. Do not interrupt someone else's job. Keep large runs, movies,
  checkpoints, and campaign reports outside tracked code (for this project,
  use the separate `vespa-runs` campaign repository without making it a dependency).
- Resume only explicitly, in the same output directory, with a trusted compatible
  checkpoint, source fingerprint, Julia/environment, and evolution settings.
  Preserve pre-resume metadata and check for repeated epochs. Never bypass the
  restart guards or deserialize untrusted Julia checkpoints. After a reboot,
  verify the process, log progress, and last valid checkpoint before relaunching.

## Documentation and Handoff

Keep instructions, examples, and tests aligned with changed interfaces. Use
Quadrant `src`, `test`, `check`, `decision`, and `notation` conventions; link an
equation to its implementation and distinguish test source from measured PASS.
For documentation changes run:

```sh
python3 -m unittest discover -s docs/scripts -p 'test_*.py'
quarto render docs --fail-if-warnings
python3 docs/scripts/check_links.py docs/_site
```

Do not commit generated HTML or unrelated regenerated evidence. End each task
with changed behavior, exact checks and their scope, run/artifact locations,
backend/precision limits, and unresolved physics. Report estimated versus
measured performance and queued versus completed work distinctly. If a gate
fails, preserve the reproducer and evidence; do not weaken it to finish the task.
