"""bench_runs: tooling for the new flat runs/ layout.

Atom: one `(source, kernel, menu, agent_features, model, replicate)` per run dir
under `benchmark-opt/runs/<run_id>/`. Each run has a metadata.json carrying a
list of measurement records (one per measurement methodology applied).

Modules:
- manifest : scan runs/ → DataFrame
- migrate  : legacy agent_bench_docker_version_*/ → runs/
- measure  : append a new measurement record to existing runs (re-profile)
- report   : generate canonical CSV from manifest filtered by agent_features +
             measurement_method
- status   : matrix.toml vs manifest diff (filled / needs-runs / needs-measure)
- retag    : bulk rewrites of agent_features / measurement-method tags
"""
