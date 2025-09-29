# MLnextstep

End-to-end log analysis, feature engineering, and machine learning pipeline for cybersecurity and system logs.

This repository contains a modular pipeline that parses raw log files, performs cleaning and preprocessing, generates feature vectors, scales those features, trains clustering and anomaly-detection models, and produces model outputs and validation summaries.

## Goals

- Provide a reproducible, end-to-end workflow for turning raw system/service logs into ML-ready feature vectors and model outputs.
- Support a variety of log formats with robust text preprocessing and validation.
- Produce interpretable model outputs (clusters, anomaly scores) and save artifacts for downstream analysis.
- Make the pipeline easy to run (single `main_pipeline.py`) while still allowing individual step execution for development.

## Project structure

Top-level tree (important files and folders):

```
MLnextstep/
├── .venv/                         # Optional local virtual environment (not committed)
├── drain3.ini                     # drain3 configuration used by the parser
├── drain3_state.bin               # optional parser state
├── log_parser_drain3_v3.py        # parse raw logs -> parsed CSV/JSON in oplogs/
├── logs/                          # raw input logs (.log files)
├── oplogs/                        # intermediate and final pipeline outputs
│   ├── csv/                       # parsed CSVs from parser
│   ├── json/                      # parsed JSON from parser
│   ├── errors/                    # parsing/processing errors saved as JSON
│   ├── cleaned/                   # normalized cleaned logs (CSV / Parquet)
│   ├── features/                  # generated feature vectors
│   ├── scaled_features/           # scaled / encoded features
│   ├── models/                     # persisted trained models (.joblib)
│   └── model_outputs/             # model outputs, performance metrics, plots
├── preprocessing/                 # additional cleaning & text processing
│   ├── log_cleaner_v1.py          # reads parsed CSVs -> cleaned CSV/Parquet
│   └── text_preprocessor_v3.py    # helper used by parser / cleaner
├── feature_engineering.py         # generate time-windowed feature vectors
├── feature_scaling.py             # validate/select, scale and encode features
├── ml_model_training.py           # train clustering/anomaly models & save outputs
├── validation/                    # validation utilities
│   └── field_validator_v3.py      # structural field validation for entries/dataframes
├── utils/                         # helper modules
│   ├── config_manager_v3.py       # configuration helpers
│   └── error_handler_v3.py        # centralized error handling
├── results/                       # final results & summaries
│   └── results.csv                # summary rows appended by pipeline
├── main_pipeline.py               # orchestrator: runs all steps end-to-end
└── README.md                      # this file
```

Brief file/folder explanations

- `log_parser_drain3_v3.py`: Main parser that reads raw logs (`logs/*.log`), applies text preprocessing (`preprocessing/text_preprocessor_v3.py`), validates entries, and writes parsed CSV/JSON/ errors to `oplogs/`.
- `preprocessing/log_cleaner_v1.py`: Loads the parser's CSVs (`oplogs/csv/*.csv`), performs normalization (timestamps, levels, IP validation, deduplication), and writes cleaned datasets to `oplogs/cleaned/` (CSV + Parquet). Returns a cleaned pandas DataFrame when called programmatically.
- `preprocessing/text_preprocessor_v3.py`: Encoding-aware file reader and line-level text normalization helpers used by the parser and cleaner.
- `feature_engineering.py`: Loads cleaned logs (`oplogs/cleaned/`), constructs time-windowed feature vectors (event counts, level ratios, entropy on IP/service/tags, temporal features, etc.), normalizes and saves feature vectors to `oplogs/features/`.
- `feature_scaling.py`: Loads feature vectors, validates and selects features, imputes missing values, scales numeric features and encodes categoricals, saves scaled features to `oplogs/scaled_features/`.
- `ml_model_training.py`: Loads scaled features and trains models (DBSCAN clustering, IsolationForest anomaly detection, PCA for visualization), saves trained models to `oplogs/models/`, writes model outputs and performance metrics to `oplogs/model_outputs/`, and creates optional plots.
- `validation/field_validator_v3.py`: Validates individual parsed entries and full DataFrames (timestamp conversion, required columns, IP checks).
- `utils/`: Small utilities for configuration and error handling used across the pipeline.
- `main_pipeline.py`: New orchestrator script that runs Steps 1–6 in sequence using the modules above. Logs progress and execution time for each step.

## Installation

Recommended Python version: 3.10+ (project was developed and tested with 3.10–3.11).

1. Create and activate a virtual environment (PowerShell commands shown):

```powershell
python -m venv .venv
# Activate in PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Install required packages. If `requirements.txt` exists, use:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the main dependencies used by the project:

```powershell
pip install pandas numpy scipy scikit-learn joblib pyarrow chardet python-dateutil matplotlib seaborn
```

Notes:
- `pyarrow` (or `fastparquet`) is used for Parquet read/write. If you prefer `fastparquet` install it instead.
- Visualization (matplotlib, seaborn) is optional but recommended for plots.

## Usage

Run the entire pipeline (recommended):

```powershell
python .\main_pipeline.py
```

What the full pipeline does (high-level):

- Step 1: Parse raw logs in `logs/*.log` and write parsed CSV/JSON to `oplogs/csv` and `oplogs/json`.
- Step 2: Clean and normalize parsed logs; cleaned datasets written to `oplogs/cleaned/` (CSV + Parquet).
- Step 3: Generate feature vectors and save to `oplogs/features/`.
- Step 4: Validate/select and scale features, save to `oplogs/scaled_features/`.
- Step 5: Train models (DBSCAN, IsolationForest, PCA), save models to `oplogs/models/` and model outputs to `oplogs/model_outputs/` (including `performance/` and `plots/`).
- Step 6: Run validation and append a summary row to `results/results.csv`.

Run individual steps (for development)

- Parse logs only:

```powershell
python .\log_parser_drain3_v3.py
```

- Clean parsed CSVs:

```powershell
python .\preprocessing\log_cleaner_v1.py
```

- Generate features:

```powershell
python .\feature_engineering.py
```

- Scale features:

```powershell
python .\feature_scaling.py
```

- Train models:

```powershell
python .\ml_model_training.py
```

Note: Running modules directly uses the file paths and defaults inside each script (e.g., `oplogs/` locations). `main_pipeline.py` calls core functions where possible to pass DataFrames in memory for speed.

Expected input formats

- The parser accepts typical system/service log lines. The code contains timestamp regexes and heuristics targeting Windows, Apache, Android, and generic formats. Place raw logs in the `logs/` folder as `.log` files.

Primary output locations

- Parsed CSV/JSON: `oplogs/csv/`, `oplogs/json/`
- Parsing errors: `oplogs/errors/`
- Cleaned normalized logs (CSV/Parquet): `oplogs/cleaned/`
- Feature vectors (CSV/Parquet): `oplogs/features/`
- Scaled/encoded features: `oplogs/scaled_features/`
- Trained models (.joblib): `oplogs/models/`
- Model outputs, performance metrics and plots: `oplogs/model_outputs/` (subfolders `performance/`, `plots/`)
- Final pipeline summary rows: `results/results.csv`

## Features and capabilities

- Robust, encoding-aware text preprocessing (via `preprocessing/text_preprocessor_v3.py`).
- Parser with per-line validation and error capture (`log_parser_drain3_v3.py`).
- Cleaning and normalization pipeline: timestamp parsing, timezone-aware datetimes, log-level normalization, IP validation, deduplication, derived time columns (`preprocessing/log_cleaner_v1.py`).
- Time-windowed feature engineering: counts, ratios, entropy measures, burstiness, ports, service stats, tag stats (`feature_engineering.py`).
- Feature validation, selection, imputation, scaling and categorical encoding (`feature_scaling.py`).
- Model training and analysis: DBSCAN clustering, Isolation Forest anomaly detection, PCA for visualization, model and result persistence, performance metrics export (`ml_model_training.py`).
- Orchestration script `main_pipeline.py` to run all steps end-to-end with logging and timing.

## Example workflow (step-by-step)

1. Put your raw logs into `logs/` (e.g., `Linux_2k.log`).
2. Run the pipeline:

```powershell
python .\main_pipeline.py
```

3. Check parser outputs:
- `oplogs/csv/<logfile>.csv` — parsed rows
- `oplogs/json/<logfile>.json` — parsed JSON newline records

4. Check cleaned data:
- `oplogs/cleaned/cleaned.parquet` and `oplogs/cleaned/cleaned.csv`

5. Check feature vectors:
- `oplogs/features/feature_vectors.parquet` (and CSV)

6. Check scaled features:
- `oplogs/scaled_features/feature_vectors_scaled.parquet` (and CSV)

7. Check ML outputs:
- `oplogs/models/` — saved .joblib models
- `oplogs/model_outputs/` — combined outputs
- `oplogs/model_outputs/performance/` — performance metrics and separate model outputs
- `oplogs/model_outputs/plots/` — PNG visualizations (if plotting libraries available)

Sample output preview (feature vectors, truncated):

```
window_start | event_count | count_error | error_ratio | unique_ip_src | service_entropy | mean_inter_event_time | ...
-------------|-------------|-------------|-------------|----------------|-----------------|-----------------------|-----
2025-09-01   |  12         |  2          | 0.1667      | 3              | 0.98            |  12.5                 | ...
```

## Logging and error handling

- Global pipeline log: `pipeline.log` in the project root (created/appended by `main_pipeline.py`). It captures step-level INFO and ERROR messages and stack traces.
- Each module configures its own logger (stdout). Many modules also write outputs and error files into `oplogs/errors/`.
- Error handling strategy:
	- The parser collects per-line errors and writes them to `oplogs/errors/` so you can inspect problematic lines.
	- The cleaner and feature modules log warnings for missing columns or parsing issues and attempt safe fallbacks (imputation, type coercion).
	- `main_pipeline.py` wraps each step with try/except and logs full tracebacks for unexpected exceptions. Critical failures halt the pipeline with a non-zero exit code.

Debugging tips

- Inspect `pipeline.log` for stack traces.
- Inspect per-module saved error files in `oplogs/errors/`.
- Run individual modules (e.g., `preprocessing/log_cleaner_v1.py`) to reproduce and isolate failures.
- Add small sample logs to `logs/` and iterate interactively to speed debugging.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository and create a feature branch.
2. Run existing unit/functional checks (if added) and test on a small dataset.
3. Follow existing code style (PEP8-ish) and include type hints where convenient.
4. Submit a pull request describing the change, tests, and a short example of how to run it.

Coding standards & Tips

- Keep functions small and testable.
- Prefer returning DataFrames from modules (instead of only writing files) to facilitate composition and unit testing.
- Add or update `requirements.txt` when adding new dependencies.

## License

Specify your license here. Example (choose one):

- MIT License — open-source and permissive. Or
- All rights reserved — if you prefer not to open-source this project.

If you want me to add a `LICENSE` file with a specific license, tell me which one and I will add it.

## Contact

- Author: (add your name or GitHub handle)
- Repository: (add GitHub repo URL if available)
- Email: (optional contact email)

----

If you'd like, I can also:
- Add a `requirements.txt` automatically by scanning the code for imports.
- Add a small sample `logs/sample.log` and a smoke-test that runs the first two steps on a tiny dataset.
- Add a `LICENSE` file for a chosen open-source license.

Tell me which of the above you'd like me to do next.
