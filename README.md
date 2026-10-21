# PMFGLM.jl

Primordial magnetic field simulations on one CPU or GPU node, written in Julia
with KernelAbstractions.jl. Includes GLM MHD, source-aware Compton drag, evolved
photon moments, full primordial mixing chemistry, periodic particle gravity,
matched magnetic initial conditions, spectra, PDFs, and independent linear
reference calculations.

The four small physics packages are bundled under `lib/`. There is no Vespa,
Enzo, CICASS, private registry, or sibling-checkout requirement. Julia packages
such as KernelAbstractions, FFTW, Metal, and CUDA resolve through the public
Julia registry. Python analysis uses NumPy, SciPy, and Matplotlib.

**Documentation:** <https://yipihey.github.io/pmf-glm/>. The Quadrant/Quarto site
includes equations, implementation, GPU use, measured test results, science
validation boundaries, restart procedures, and collaborator onboarding. The
site source lives in `docs/`; deployment runs through GitHub Actions.

## First Run

Install Julia 1.12 or newer, then:

```sh
git clone https://github.com/yipihey/pmf-glm.git
cd pmf-glm
julia --project=. --threads=auto scripts/bootstrap.jl --backend=cpu
julia --project=. --threads=auto scripts/run_pmf.jl examples/smoke.toml --output=results/first-cpu
```

The first call compiles Julia kernels. Results, resolved configuration, source
hash, machine identity, startup/phase timing, and log are in the selected run
directory. `tail -f results/first-cpu/run.log` follows progress. A short
full-physics installation check is `examples/cosmology-smoke.toml`.

For Apple Silicon:

```sh
julia --project=. --threads=4 scripts/bootstrap.jl --backend=metal
julia --project=environments/metal --threads=4 scripts/test_gpu.jl --backend=metal
julia --project=environments/metal --threads=4 scripts/run_pmf.jl examples/cosmology-smoke.toml --backend=metal --output=results/first-metal
```

For NVIDIA GPUs on Linux:

```sh
julia --project=. --threads=auto scripts/bootstrap.jl --backend=cuda
julia --project=environments/cuda --threads=auto scripts/test_gpu.jl --backend=cuda
julia --project=environments/cuda --threads=auto scripts/run_pmf.jl examples/cosmology-smoke.toml --backend=cuda --output=results/first-cuda
```

Metal uses Float32 and MPS real FFTs. CUDA uses Float32 production kernels and
cuFFT. CPU reference runs support Float32/Float64, threaded KA kernels, and
FFTW. `--threads=auto` enables CPU workers; `PMF_FFTW_THREADS=4` sets FFTW's
separate thread count (default at most four). GPU tests require real hardware
and fail explicitly if the requested device is unavailable. CUDA validation
must be run on an NVIDIA node; availability of an extension is not a measured
accuracy or performance result.

## A PMF Calculation

Copy `examples/pmf-recombination.toml`, set resolution, box, starting redshift,
field strength, physical cutoff, and output epochs, then run with the same CLI.
The example has a **3 pG comoving** field (`b0_ng=0.003`), a **1 ckpc/h** box,
full chemistry with expansion, MHD CFL 0.2, and photon nonlinear CFL 0.1.
`--dry-run` prints the resolved solver settings without allocating the mesh.

```sh
julia --project=environments/metal -t 4 scripts/run_pmf.jl examples/pmf-recombination.toml --backend=metal --output=runs/pmf-3pG-N128
python3 -m pip install -e '.[test]'
pmf-analyze runs/pmf-3pG-N128
```

Check the analysis tool's `--help` for plotting/output options. Large field dumps
are opt-in; inline tables are the default. Standard outputs include magnetic,
baryon/DM density, photon and slip spectra, magnetic-pressure/tension force
spectra, density PDFs, mean ionization, and energy ledgers.

## Tests and Benchmarks

```sh
julia --project=. --threads=4 scripts/validate.jl
python3 -m pytest
julia --project=environments/metal --threads=4 scripts/validate.jl --backend=metal
julia --project=environments/metal --threads=4 scripts/benchmark.jl --backend=metal --n=128
```

Measured results and source fingerprints feed the documentation. The numerical
suite covers smooth-wave convergence, shocks, field-loop advection, Orszag-Tang,
divergence-free and mode-locked ICs, stiff drag, photon response, gravity,
species transport, RECFAST recombination, and GPU/CPU agreement. Test results
from the original Vespa campaign are historical evidence, and are labelled
separately from this extraction's tests.

The independent `tools/reference` models reproduce linear Schiff-style
solutions and generate matched handoffs. The production photon closure uses
P1 moments. Native nonlinear second-order nonlocal Ly-alpha transport and the
full unequal-time visibility calculation are not implied by a linear-mode
test. UV cutoff and starting redshift are physical sensitivity parameters.
See the documentation before interpreting a science campaign.

Run the complete native mean-field comparison, including a correlated handoff,
the Julia production solver, and an independent complex-mode assessment:

```sh
uv run pmf-compare-nonlinear results/matched-metal --backend=metal \
  --resolution=32 --zend=3190 --execute
```

Use `--zend=500 --resolution=64` for a longer recombination comparison and
converge resolution/timestep independently. A nonlocal Ly-alpha response table
is an explicit optional `--response-table`; the default compares local full
chemistry. The output contains its actual run metadata, oracle hashes, numerical
gate, PDF/PNG comparison, and unfitted complex amplitudes. Passing a short
interval does not establish a converged long-interval result.

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md), start with a small CPU case, and attach
the configuration, source/manifest hashes, test evidence, and backend details
to numerical changes. Coding agents should also follow [AGENTS.md](AGENTS.md).
Keep large simulation outputs outside the code repo.
The extraction history is in `provenance/extraction.json`; bundled package
licenses and upstream attribution are retained in [NOTICE.md](NOTICE.md).
