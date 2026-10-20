# Provenance and Attribution

PMFGLM was extracted from the working PMF solver in Tom Abel's Vespa.jl and
the associated vespa-runs validation/analysis tools. The upstream commit and
per-file SHA256 at extraction are in `provenance/extraction.json`. Modified
working files were intentionally included; a commit alone does not identify
the source. Local portability fixes are documented by this repository's Git
history and the validation source fingerprints.

ChemistryKernels and EmissionKernels retain the Enzo public license notices in
their bundled `LICENSE` files. Those notices cover the original Enzo chemistry
and radiative rate lineage, including the University of Illinois/NCSA and
University of California license terms. They remain applicable to those
components. PMFGLM's new integration and documentation use the root BSD license.

The inherited `recfast_v2_xe.csv` fixture is generated with CAMB/RECFAST-v2 using
the cosmological parameters recorded in its header; `gen_recfast_v2.py` records
the generation procedure. This small numerical fixture is test data rather
than a substitute for an independent recombination-code installation.

Quadrant's Quarto extension/styles are from yipihey/quadrant; its MIT notice is
retained in `docs/_extensions/yipihey/quadrant/`.

The project-owned MHD/Poisson and independent reference implementations are
distributed here under the root BSD-3-Clause license. The independent
Schiff-style reference modules are from the vespa-runs
implementation, based on the equations and corrected manuscript/reference
choices discussed with Schiff and Venumadhav. They are not a claim to ship
the authors' repository or grant a license to external data/manuscripts.
See `tools/reference/provenance.json` and the method references in the site.

Please cite the numerical/physical methods used by a calculation as well as
this software. `CITATION.cff` supplies the repository citation without inventing
a publication or archive DOI.
