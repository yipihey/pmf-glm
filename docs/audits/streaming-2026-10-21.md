# Baryon–DM velocity and 2.5D audit — 21 October 2026

This original audit records the work before velocity-fix integration. Historical
measurements below retain their original scope; the follow-up integration
section records the combined implementation and refreshed evidence.

## Repository baseline

Before edits, `git fetch origin` succeeded, the worktree was clean, and both
local `HEAD` and GitHub `origin/main` were
`d446dd310f4e91161c1889896500ded7ae3687a9`.
The available history and source were inspected before implementation; there
was no separate streaming/2.5D implementation in another local or remote branch.
No commits or pushes were made during the original audit.

## Existing capabilities and findings

| Area | Existing implementation | Audit result |
|---|---|---|
| Gas velocities | Conserved momentum divided by density; Jedamzik physical velocity factor and Hubble momentum decay | Consistent with particle velocities after separate unit conversions; common-force and free-decay tests added |
| DM velocities | Supercomoving `dx/dtau`, lattice displacements or absolute periodic positions; coupled Poisson gravity | Evolution convention is appropriate; physical diagnostic conversion incorrectly used the spectral box instead of the physical box |
| Mode diagnostics | Gas code velocity; one DM physical velocity component on original lattice labels | Added physical gas rows and all three DM components; explicitly documented Lagrangian/Eulerian registration limits and corrected physical wavenumbers |
| Photon diagnostics | Inverse transform of the live evolved photon state | **Physical side effect found:** FFTW c2r overwrote live modes. Added scratch-copy inversion and tests for unchanged state and later evolution |
| Expansion constants | Exact physical box conversion but rounded Mpc constant in two Hubble/clock functions | Unified constants; gas and DM gravitational impulses now pass the strict common-unit identity |
| Particle step bound | Difference of nearly equal square roots | Rationalized the expression to retain a finite displacement bound for fast streams and tiny accelerations |
| Streaming background | Uniform gas/DM default ICs; no named coherent streaming option or general relative-velocity history | Added physical km/s boost at the actual start epoch, DM rest frame, sign `vb-vc`, and current-particle-position sampling |
| Planar evolution | 3D MHD vectors and cubic coupled solver | Added zero-wavenumber-axis magnetic initialization with all vector components retained; pure MHD and coupled symmetry checks |
| Exploration | Central gas density slices, maximum projections, PNG/PDF renderer, documented external FFmpeg workflow | Added named frame configuration, separate DM frames, direct GIF encoding, input hashes, cycle ordering, duplicate rejection and fractional-redshift filename correction |

Review dimensions: correctness had significant diagnostic defects, now covered by
regressions; performance retains the cubic coupled solver and adds only opt-in
work at diagnostic cadence; maintainability adds named TOML options, a shared
CPU/device test helper and documented interfaces. No new remote service,
checkpoint format, or untrusted-deserialization path was introduced.

## How to use it

Start with [the small example](../../examples/streaming-planar-smoke.toml),
[the output/animation workflow](../outputs.qmd#sec-streaming-output), and
[the velocity conventions](../physics.qmd#sec-streaming). Its short redshift
interval is an installation/capability check, not a recombination campaign.

`streaming.tsv` records physical peculiar cm/s, with equal-DM-mass weighting
at current particle positions. `pmf-analyze` discovers this table. Interpolating
gas velocity to the moved particles avoids subtracting gas-grid modes from DM
modes indexed by the original lattice. The subsequent velocity correction
replaced DM velocity label modes with Eulerian CIC velocity estimates and an
empty-cell fraction; displacement modes retain their lattice-label meaning.

## Scientific and performance boundaries

- A supplied constant relative velocity is a local coherent background, not a
  cosmological random streaming realization. Its value is at `zstart`; no earlier
  evolution or normalization epoch is invented. Photon initial bulk motion follows
  the declared photon-comoving option.
- Built-in PMF ICs start with uniform gas and DM density. They generate PMF-driven
  DM fluctuations, not a primordial adiabatic transfer realization. A cosmological
  streaming suppression study needs correlated density/velocity/thermal/photon ICs,
  matched zero-stream controls, complex-mode checks, and separate temporal/spatial
  convergence. Existing arbitrary correlated imports remain the handoff route.
- The independent matter-era moving-background reference follows
  [Tseliakhovich & Hirata (2010)](https://arxiv.org/abs/1005.2416) in the DM frame,
  with explicitly simplified expansion/thermal physics. It is `REFERENCE_ONLY`;
  its checks are not nonlinear-solver or halo-abundance certification.
- `magnetic.invariant_axis=1|2|3` is a genuine planar initial condition with three
  vector components, evolved on a replicated cube. Storage/update cost remains
  **O(N^3)**. A reduced-memory N-by-N-by-1 coupled backend is not implemented and
  no dimensional-reduction speedup was measured.
- Relative-velocity histories are particle-mass weighted, not Eulerian relative
  velocity spectra. A multistream/empty-cell estimator for such spectra remains
  separate work. Planar shell spectra are not isotropic 3D cosmological spectra.
- CUDA is `NOT_RUN` in this audit. CPU Float32/Float64 and actual Metal Float32
  are the tested scopes. Small completed examples and cycle-limited parity tests
  do not validate a long recombination trajectory or every streaming observable.

## Reproducers and retained negative results

The original failed smoke is retained under `results/streaming-audit-cpu-f32`.
Pre-fix uniform-stream controls are in `results/streaming-audit-uniform-*`.
The pre-fix coupled planar CPU failure is in `results/production-cpu-4AsIkZ`;
its log and the Metal reference-integrator compiler failure are retained under
`results/streaming-audit-failures`. The unsupported portable Metal reference
integrator was not changed; the successful device gate uses the production cube
integrator. No numerical tolerance was relaxed to remove the planar failure.

A manual checkpoint-inspection command was rejected by automatic approval review
because it treated deserialization as bypassing restart guards. The diagnosis was
completed using newly allocated arrays and raw diagnostics; checkpoint comparisons
used the repository's production validator on its own newly generated runs.

## Original audit validation and artifacts

Commands were run from the repository root with Julia 1.12.6 and four Julia threads.

| Check | Outcome and scope | Evidence |
|---|---|---|
| `julia --project=. -t 4 test/runtests.jl` | Passed, including CPU Float32/Float64 streaming and photon-state regressions | `results/streaming-audit-failures/julia-final.log` |
| `uv run --offline python -m pytest` | 116 passed; independent references, analysis and GIFs | `results/streaming-audit-failures/python-final.log` |
| `julia --project=environments/metal -t 4 scripts/test_gpu.jl --backend=metal` | 312 passed on the real Metal device, Float32, production cube integrator | `results/streaming-audit-failures/metal-final.log` |
| `scripts/validate_production.jl --backend=cpu` in its matching project | PASS; completed endpoint smoke; four-cycle planar restart/cadence comparisons | [latest CPU record; see integration below](../data/production-cpu.toml) |
| `scripts/validate_production.jl --backend=metal` in its matching project | PASS; completed endpoint smoke; four-cycle planar restart/cadence comparisons | [latest Metal record; see integration below](../data/production-metal.toml) |
| `uv run --offline python scripts/check_portability.py` | Passed | Internal active-root sources and runtime paths |
| `python3 -m unittest discover -s docs/scripts -p 'test_*.py'` | 6 passed | Documentation preparation/link unit checks |
| `quarto render docs --fail-if-warnings`; `python3 docs/scripts/check_links.py docs/_site` | Passed | Rendered source, asset and fragment links |

The original unexpanded production records remain recoverable in Git history;
only the CPU/Metal production records were refreshed by the validator.

The following values are the historical audit measurements, not the refreshed
combined-source production records linked above.

| Backend | Planar maximum scaled error | Restart maximum scaled error | Diagnostic-cadence maximum scaled error |
|---|---:|---:|---:|
| cpu Float32 | 1.70779927e-07 | 1.7077993e-07 | 0 |
| metal Float32 | 0 | 3.55271368e-15 | 0 |

All three production scaled-error limits are `5e-6`. These use absolute
normalization floors and are regression checks, not relative-error convergence
claims for arbitrarily tiny density modes.

The separate CPU Float64 example is **COMPLETED** at the requested
`z=3199.9` after starting at `z=3200`. Its raw metadata, resolved settings,
histories, spectra, density PDFs and frame manifests are under
`results/streaming-audit-cpu-f64-final`. Gas and DM GIFs are in `animation/gas.gif` and
`animation-dm/dm.gif`, with rendered analyses under `analysis/`. The DM signals
in this very short run are tiny; it is a capability check.

Measured validator wall times include compilation, subprocess startup, output
and several runs; they are not warm MHD throughput benchmarks. No speedup or
large-grid memory feasibility is inferred. Nothing remains queued by this audit.


## Integration with the velocity corrections

The streaming/planar work is included as a follow-up to velocity-only commit
`ecb1a63`. The combined Julia implementation has source fingerprint
`402403196ca5f6d958ebba5dafd0ecf4145cecbe7e378f765dd6bb618de2259c`. Both production records now refer to this combined source;
the earlier velocity-only production records remain available at their immutable
commit links in the [velocity audit](relative-velocities-2026-10-21.md).

The combined source passed:

- `julia --project=. -t 4 test/runtests.jl`, including CPU Float32/Float64 checks;
- `python -m pytest -q` in the project environment: 118 tests and three subtests;
- `julia --project=environments/metal -t 4 scripts/test_gpu.jl --backend=metal`:
  324 real-device Float32 checks;
- `scripts/validate_relative_velocities.jl results/velocity-combined-cpu-f64 f64`:
  26 production assertions, with fixed ionization and the nonzero-force cap study;
- the matching-source CPU production gate, and a fresh Metal production gate,
  using `scripts/validate_production.jl --backend=cpu|metal` in the matching project;
- portability, six documentation unit checks, warning-free Quarto rendering,
  and rendered local-link checks.

| Backend | Planar maximum scaled error | Restart maximum scaled error | Diagnostic-cadence maximum scaled error |
|---|---:|---:|---:|
| cpu Float32 | 0 | 1.10738752e-10 | 0 |
| metal Float32 | 0 | 2.77555756e-17 | 0 |

The CPU production record is reused from `results/production-cpu-tH3U4i` only
after checking that its source fingerprint exactly matches the combined source.
The fresh Metal record points to `results/production-metal-hCD4xG`. The endpoint
smokes are COMPLETED; the four-cycle restart/cadence controls are deliberately
STOPPED_AT_LIMIT and are not long-time physical convergence studies.

The example was dry-run and then executed with:

```sh
julia --project=. -t 4 scripts/run_pmf.jl examples/streaming-planar-smoke.toml --backend=cpu --output=results/streaming-combined-smoke --dry-run
julia --project=. -t 4 scripts/run_pmf.jl examples/streaming-planar-smoke.toml --backend=cpu --output=results/streaming-combined-smoke
```

Its actual final redshift is 3199.9 after eight cycles, with status COMPLETED.
`pmf-analyze results/streaming-combined-smoke` produced the analysis and raw
reductions. The frame renderer produced three-frame gas and DM GIFs under
`animation/gas.gif` and `animation-dm/dm.gif`, with fixed limits and payload hashes.
Current validation/analysis logs are retained under
`results/streaming-combined-validation-logs`. The GIF dependency is explicitly
specified as `Pillow>=9.1`; offline environment resolution and installation pass.

CUDA remains NOT_RUN. These short examples and parity checks do not establish
cosmological transfer initial conditions, a recombination streaming prediction,
spatial convergence, or reduced-memory planar evolution. No throughput speedup
or large-grid memory feasibility is claimed.
