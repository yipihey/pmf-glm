# Measured evidence only

The documentation pre-render accepts `validation*.json`, `validation*.toml`,
`benchmark*.json`, and `benchmark*.toml` here. Files are optional; an absent
measurement renders as **not yet measured**, never as a pass or zero timing.
Both JSON and TOML use only the Python standard library (Python 3.11+).

Preferred records contain a top-level `schema_version`, `generated_at`, `commit`,
`backend`, `julia_version`, `hardware`, `status`, and a `tests` or `benchmarks`
array. Each item should include `name`, `status`, and the measured quantities,
tolerances, units, and exclusions that support its conclusion. The renderer
also accepts producer-specific schemas and shows the actual top-level metadata
and individual records without silently inferring a pass.

Keep raw evidence (including failures and skips) with its logs and configuration.
Publishing a file does not make it current: compare its revision to the site
revision. The prepare script does not execute Julia, regenerate measurements,
or upgrade an upstream campaign into extraction certification.
