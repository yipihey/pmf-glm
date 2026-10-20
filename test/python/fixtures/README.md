# Small Offline Fixtures

`recfast_v2_xe.csv` is copied byte-for-byte from
`ChemistryKernels.jl/test/fixtures/recfast_v2_xe.csv`. Its header records the
CAMB/RECFAST-v2 cosmology and Gaussian corrections. The original generator was
inspected, not imported: CAMB is **not** required to run these tests. The file
contains total electrons per hydrogen nucleus, including helium at high redshift;
the hydrogen-only comparison is restricted to the neutral-helium window.
`licenses/ChemistryKernels.LICENSE` retains its source project's license verbatim.
See `tools/reference/provenance.json` for original absolute paths and SHA-256s.

`history.tsv`, `spectra.tsv`, and `density_pdf.tsv` are newly authored synthetic
fixtures matching the actual `pmf_dm_lattice_bench.jl` writer schemas. They are not
simulation measurements. They exercise initial `measured=0` rows, disabled
chemistry NaNs, empty spectral bins, unequal angular mode counts, and independent
pressure/tension/Lorentz powers. Their box coordinate length is one; use
`--length-unit box --box-ckpc-h 1 --hubble 0.674` when plotting them.
`metadata.toml` is a synthetic example of the root runner's physical-box metadata,
used to verify unit inference and input/output separation.
