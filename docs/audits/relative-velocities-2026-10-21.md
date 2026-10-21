# Relative velocity patch review and validation — 21 October 2026

This record describes the velocity-only commit `ecb1a63`. The subsequent
[streaming/planar integration](streaming-2026-10-21.md) has separate combined-source
validation. Numerical tables here retain their recorded source fingerprints.

The revised patch was prepared against GitHub `main` at
`d446dd310f4e91161c1889896500ded7ae3687a9` in an isolated checkout. The existing
uncommitted streaming/planar feature work was preserved separately and is not
part of this change. The related photon diagnostic scratch-copy correction is
included because output must not alter persistent photon modes.

## Changes

- Correct photon dipole expansion in Jedamzik units, preserving the monopole.
- Fingerprint corrected velocity transport and the gravity/drag cap in checkpoints.
- Limit the outer gravity/drag splitting impulse using the fast slip eigenvalue.
- Deposit all three particle velocity components at current Eulerian positions,
  with physical-box units and an explicit empty-cell fraction.
- Reuse two MHD integrator scratch buffers after force-mode reductions; complete
  deposition and mode reduction before reusing them for another component.
- Copy photon modes to reusable FFT scratch before diagnostic inverse transforms.
- Add CPU Float32/Float64 and real-device checks, a reproducible allocation
  benchmark, and nonzero-force production temporal convergence.

## Measured velocity convergence

The production validator starts a displaced periodic DM lattice from rest, with
uniform gas and zero photon velocity. Nonzero Poisson gravity generates a gas
velocity mode. At the common endpoint z=1199.5, it compares the signed complex
baryon–photon slip while varying only the outer gravity/drag cap. The reference
cap is 0.0025 on the same N=8 grid. The predeclared signal floor is 0.001 cm/s;
the error must decrease by at least 20% per refinement and be below 2% at 0.01.

| Cap | CPU Float32 relative complex error | CPU Float64 relative complex error |
|---|---:|---:|
| 0.08 | 3.2510% | 3.2510% |
| 0.04 | 1.6806% | 1.6805% |
| 0.01 | 0.3534% | 0.3534% |

Each precision passed 26 production assertions across homogeneous transport,
independent SciPy Radau comparison, and nonzero-force temporal refinement.
The homogeneous cases reached z=600; the gravity cases reached z=1199.5.
Raw values and source/oracle fingerprints are retained in
[measurements](../data/relative-velocities/measurements.json),
[Float32 gravity modes](../data/relative-velocities/gravity-f32.tsv),
[Float64 gravity modes](../data/relative-velocities/gravity-f64.tsv),
and the [Float32](../data/relative-velocities/validation-f32.log) and
[Float64](../data/relative-velocities/validation-f64.log) validator logs.
These are fixed-ionization, chemistry-off controlled cases. The fine trajectory
is a temporal reference, not an independent nonlinear gravity oracle or spatial
convergence result.

## Allocation measurement

Julia 1.12.6, CPU Float32, four Julia threads, `PMF_FFTW_THREADS=4`, one velocity
component, minimum of three warm calls after two warmups. The benchmark includes
mode reductions and TSV output, and excludes state allocation. Host allocations
are measured with `@allocated`; this is not a GPU peak-memory measurement.

| N | Submitted patch bytes/call | Revised bytes/call |
|---|---:|---:|
| 16 | 197936 | 164464 |
| 32 | 430896 | 164464 |
| 64 | 2295184 | 165072 |

The removed grid allocations scale with volume; the remaining overhead is
approximately constant over these grids. No throughput improvement is claimed:
the small CPU timings were similar and collected while other checks ran.
See the [before](../data/relative-velocities/allocation-before.tsv) and
[after](../data/relative-velocities/allocation-after.tsv) raw measurements.
Reproduce the revised measurement from the repository root:

```sh
julia --project=. -t 4 tools/benchmarks/velocity_diagnostics.jl src/production/runner.jl
```

## Regression scope

The exact standalone update passed the full Julia CPU regression suite,
115 Python tests with three subtests, all 255 applicable Metal Float32 backend
checks on real hardware, the portability check, six documentation unit tests,
and the warning-free Quarto build and rendered-link check. The focused relative
velocity tests contain 73 CPU assertions, including Float32/Float64 expansion,
translated deposition, repeated buffer reuse, empty cells, photon diagnostic
purity, physical units, and old-checkpoint rejection.

The production CPU gate reached its requested smoke endpoint, passed four-cycle
checkpoint/restart parity, and gave identical final fields with sparse versus
frequent diagnostics. The Metal production gate likewise reached its target,
passed restart parity (maximum scaled error 1.11e-16), and gave identical final
fields across diagnostic cadences. CPU restart maximum scaled error was
3.64e-10. Machine-generated production evidence is in
[CPU](https://github.com/yipihey/pmf-glm/blob/ecb1a6339f978557694bea41a5912674d45dbcb2/docs/data/production-cpu.toml) and
[Metal](https://github.com/yipihey/pmf-glm/blob/ecb1a6339f978557694bea41a5912674d45dbcb2/docs/data/production-metal.toml).
Read each record's source fingerprint, scope, actual status, and run directory;
the cycle-limited restart controls are not endpoint science runs.

CUDA was NOT_RUN because no NVIDIA hardware was available. This change does
not establish through-recombination convergence, force/grid convergence, a
full photon hierarchy, or cosmological streaming statistics. Large-grid GPU
memory and performance remain unmeasured.

## Development failures retained

An intermediate validator run errored because Julia's `Cmd` vector constructor
does not accept `dir`; the corrected nested constructor was rerun in fresh
output directories. An integrated-checkout restart attempt also correctly
rejected a checkpoint after its source fingerprint changed during development;
the gate was rerun after freezing the implementation, without overriding the
restart guard. Sandboxed Metal discovery exposed zero devices and Quarto could
not open its user-cache database; the final GPU and render checks ran with the
required host access. Intermediate run directories and logs were retained in
the local results archive; none of these failures was relabelled as passing.
