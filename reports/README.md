# reports/ (generated locally, not published)

Everything in this folder is produced by the pipeline:

| File | Made by |
|---|---|
| `data_feasibility.md` | written during the Phase One data audit |
| `experiments/*.json`, `*.csv` | `scripts/03`–`06` |
| `results_summary.md`, `final_research_report.md`, `cv_evidence.json`, `figures/*.png` | `scripts/07_make_report.py` |

These files contain aggregate statistics derived from Deribit market data. Deribit's terms
of service restrict market data and derived data to personal use, so the generated reports
are kept out of the public repository. To reproduce them:

```bash
uv sync
uv run python scripts/01_download_data.py
uv run python scripts/run_all.py
```

The Track A simulation outputs (`experiments/track_a_*.json`, `math_validation.json`) are
derived from simulated data only, and are just as reproducible from the fixed seeds in
`configs/experiment.toml`.
