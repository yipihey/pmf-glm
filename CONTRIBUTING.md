# Contributing

Start with the CPU bootstrap, smoke run, and `scripts/validate.jl` from the
README. Run `python3 -m pytest` after installing `.[test]` for the independent
oracle and analysis tests. Optional GPU tests run only on the requested device;
a skipped GPU job cannot certify the backend.

## Code Map

- `src/PMFGLM.jl`: small public facade; configuration/provenance orchestration.
- `src/production/runner.jl`: full coupled PMF driver, extracted from the
  production Vespa runner without changing its physical kernels.
- `lib/MHDKernels`: finite-volume GLM, stiff drag, photon moments, PMF ICs.
- `lib/PoissonKernels`: native FFT Poisson, CIC, particles, spectral diagnostics.
- `lib/ChemistryKernels`: primordial chemistry, helium, mixing, precise species storage.
- `lib/EmissionKernels`: analytic cooling and emission building blocks.
- `tools/reference`: independent Float64 linear/transfer calculations.
- `tools/analysis`: reproducible plots and comparisons from compact output tables.
- `test/`: portable physics tests, independent fixtures, and explicit GPU gates.
- `docs/`: Quadrant site, source links, validation evidence, numerical limits.

Bundled packages retain their names and UUIDs so kernel extraction remains
auditable. Edit the bundled source here, not an unrelated sibling checkout.
`scripts/vendor_sources.py` refuses to overwrite an extraction; future upstream
updates need reviewed diffs and new provenance. No submodule checkout is needed.

## Numerical Changes

Use KernelAbstractions for shared device logic and existing backend extensions
for FFT plans, allocations, and optimized launch geometry. Keep time-varying
fields on the device, reuse scratch, and only copy reduced diagnostics. GPU
benchmarks must synchronize and exclude compilation explicitly.

Use fixed physical Fourier modes for grid convergence. Keep reference-band
normalization, seed, curl symbol, mode lock, and cutoff fixed. Vary timestep and
spatial resolution independently. Include amplitude/phase error, conservation,
divergence, ionization, thermal response, and common-band spectra. State the
precision floor; a tiny derived Float32 mode is not evidence of convergence.

Check CPU Float64 when feasible, CPU Float32 for device parity, and the actual
target GPU. Do not replace a failed physical gate with a relaxed tolerance
without documenting the reason and retaining the original result. Source
fingerprints in old results are immutable observations.

## Reproducible Runs

Use a new output directory or an explicit compatible checkpoint restart.
`metadata.toml` records settings, machine, Julia/thread counts, source hash,
active project, and Manifest checksum. Archive the resolved Manifest with a
published campaign. Julia Serialization checkpoints are version-sensitive:
use the same Julia/package environment, and never load untrusted checkpoints.
Raw IC field files are little-endian Float32 arrays in Julia column-major order.

Use the configured cosmological density and radiation temperature consistently;
`b0_ng` is comoving nG and `box_ckpc_h` is comoving kpc/h. A larger starting
redshift increases physical B as `(1+z)^2`, so compare comoving spectra.

## Documentation and Review

Build with `quarto render docs`. The pre-render script generates source views,
option tables and measured test results. Use Quadrant `src`, `test`, `check`,
`decision`, and `notation` conventions. Report unmeasured CUDA/Metal results as
unmeasured. Keep the library install independent of Quarto/Python plotting.

For bug reports attach a small TOML reproducer, metadata, the last log lines,
backend/device, Julia version, and whether the issue survives timestep/grid
refinement. For performance reports state cold-start versus warm throughput
and MHD-only versus full coupled timing. A scientifically interesting negative
result belongs in the validation ledger too.
