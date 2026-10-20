# PMFGLM documentation

This is a Quarto **book**, adapted from `quadrant/template`, not a Sites project.
The source of truth is the package in the parent directory. Requires Quarto
1.7+ and Python 3.11+; no Julia or scientific Python packages are needed to render.

From the repository root:

```sh
python3 -m unittest discover -s docs/scripts -p 'test_*.py'
quarto render docs --fail-if-warnings
python3 docs/scripts/check_links.py docs/_site
quarto preview docs --no-browser
```

In a restricted macOS workspace that cannot write the ordinary Quarto cache,
prefix the render command with `HOME="$PWD/docs/.quarto/home"`. This keeps
Quarto's own cache inside the ignored workspace; it is not needed in normal CI.

`prepare.py` is a pre-render hook; it writes ignored `_variables.yml`,
`_includes/`, and local source viewers in `sources/`. It never modifies package
sources. Its `src` and `test` shortcodes verify source paths and symbol text
before rendering; test-source links do not claim that a test passed. The link
checker validates all local page/asset targets and HTML fragments, including
rendered source line anchors. External links are not fetched.

Generated evidence is optional: see [data/README.md](data/README.md). The site
renders actual `validation*`, `production*`, and `benchmark*` JSON/TOML records, marks missing
records as "not yet measured", and retains metadata for revision checks. Do not
commit generated HTML; publish `docs/_site` with the Pages workflow. A maintainer
must select GitHub Actions as the repository's Pages source separately.

## Quadrant adaptations

- Retained: Quarto book navigation, Cosmo/Darkly themes, `_brand.yml` palette,
  theme-aware `qd-*` tokens, numbered sections/equations, code tools, search,
  edit/source/issue links, Hypothesis, notation/status appendices, revision footer.
- Replaced Rust-only rustdoc/test discovery with checked Julia source viewers.
- Replaced WebAssembly laboratory experiments with explicit Julia commands and
  configuration files. There is no browser simulator and no fake lab button.
- No unverified bibliography or DOI is invented. Source lineage and the scope
  of imported science-reference tools are documented in `status.qmd`.
