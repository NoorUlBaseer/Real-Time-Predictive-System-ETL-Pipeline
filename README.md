# Real-Time Predictive System – ETL Pipeline

This repository contains the end-to-end ETL and MLOps pipeline for a **Real-Time Predictive System (RPS)** that uses **news about technology/stock markets** to build a sentiment-based signal, train a predictive model, and serve real-time predictions via an API. The project is built around **Apache Airflow (Astro Runtime)**, **DVC + DagsHub**, **MLflow**, and a **FastAPI inference service** with **Prometheus/Grafana** monitoring.

---

## High-Level Architecture

- **Data Source:** GNews API (technology/stock-related articles).
- **Orchestration:** Apache Airflow DAG `stock_news_pipeline` in `dags/stock_news_pipeline.py`.
- **Storage & Versioning:**
  - Raw JSON: `data/raw/daily_news.json`.
  - Processed CSV: `data/processed/daily_news.csv` tracked with **DVC** (remote on DagsHub S3).
- **Profiling & Data Quality:** `ydata-profiling` generates a daily HTML report at `data/reports/daily_quality_report.html`, logged to **MLflow**.
- **Modeling:** Sklearn `RandomForestRegressor` trained on engineered sentiment and time features; runs logged and registered in **MLflow Model Registry** on DagsHub.
- **Model Governance:** `scripts/compare_metrics.py` compares the latest candidate run vs the current champion and promotes only if metrics improve.
- **Serving:** `inference/app.py` exposes a **FastAPI** endpoint for predictions and integrates **Prometheus** metrics.
- **Monitoring:** `docker-compose.monitoring.yml` spins up the prediction API, Prometheus, and Grafana.

---

## Repository Structure

Key paths:

- `dags/stock_news_pipeline.py` – Main Airflow DAG for the daily ETL + training pipeline.
- `data/raw/` – Raw GNews JSON dumps.
- `data/processed/` – Cleaned and enriched CSV data (`daily_news.csv` + `.dvc`).
- `data/reports/` – Generated data quality/profile reports (HTML).
- `inference/app.py` – FastAPI app for online inference + monitoring.
- `scripts/trigger_and_monitor.py` – Utility to trigger and monitor the DAG via Airflow REST API.
- `scripts/compare_metrics.py` – MLflow/DagsHub-based model comparison and champion selection.
- `docker-compose.override.yml` – Volume mounts for Airflow runtime (data, `.dvc`, `.git`).
- `docker-compose.monitoring.yml` – Stack for API + Prometheus + Grafana.
- `requirements.txt` – Python dependencies for the ETL and training environment.
- `Dockerfile` – Astro runtime base image for Airflow.

---

## Airflow DAG: `stock_news_pipeline`

The DAG orchestrates the following tasks:

1. **`extract_live_data`**
   - Calls GNews API for the previous day’s news.
   - Performs basic quality checks (HTTP status, non-empty articles, required fields).
   - Writes validated articles to `data/raw/daily_news.json`.

2. **`pull_dvc_history`**
   - Clones the GitHub repo (branch `dev`) to a temp directory.
   - Uses **DVC** with a DagsHub S3 remote to pull historical `data/processed/daily_news.csv`.

3. **`transform_and_profile`**
   - Converts raw articles into a Pandas DataFrame.
   - Adds features:
     - `publishedAt` (parsed datetime).
     - `hour_of_day`, `day_of_week`.
     - `source_name`.
     - Sentiment scores from `TextBlob` for `title`, `description`, and `content`.
   - Drops unused columns and generates a `ydata-profiling` minimal report.
   - Logs profiling report and metrics (avg sentiment, row count, unique sources) to **MLflow**.
   - Merges with historical data (if any), deduplicates on title + publish time, and saves updated `daily_news.csv`.
   - Skips downstream steps if no new unique data is found (via `AirflowSkipException`).

4. **`dvc_add_and_push`**
   - Re-initializes a local DVC repo (no SCM), adds `data/processed/daily_news.csv`, and pushes to the DagsHub S3 remote.
   - Returns the generated `.dvc` file contents.

5. **`train_model`**
   - Reads `data/processed/daily_news.csv`.
   - Builds a synthetic target `market_change` as a function of sentiment features (demo setup).
   - Trains a `RandomForestRegressor` on:
     - `hour_of_day`, `day_of_week`, `title_sentiment`, `desc_sentiment`, `content_sentiment`.
   - Logs parameters, metrics (RMSE, MAE, R²), and the model artifact to **MLflow**.
   - Attempts to register/attach the run to the `Stock_Sentiment_Predictor` model in MLflow Model Registry.

6. **`git_commit_and_push`**
   - Clones the GitHub repo on branch `dev` to a temp directory.
   - Writes the `.dvc` file into the correct path.
   - Commits and pushes the updated data version to GitHub.

The DAG is defined with manual scheduling (`schedule=None`) and is intended to be triggered daily via a GitHub Actions workflow in the repository.

---

## Inference Service (FastAPI)

File: `inference/app.py`

- Loads the trained model from `model_dir/stock_sentiment_model.pkl`.
- Exposes:
  - `GET /` – Health check (includes whether the model is loaded).
  - `POST /predict` – Takes JSON with:
    - `hour_of_day` (int)
    - `day_of_week` (int)
    - `title_sentiment` (float)
    - `desc_sentiment` (float)
    - `content_sentiment` (float)
- Uses `prometheus_fastapi_instrumentator` and `prometheus_client` to expose metrics:
  - `total_predictions` (Counter)
  - `drifted_predictions` (Counter)
  - `data_drift_ratio` (Gauge)
- Simple drift detection: flags requests where `abs(title_sentiment) > 0.95`.

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour_of_day": 10,
    "day_of_week": 1,
    "title_sentiment": 0.2,
    "desc_sentiment": 0.1,
    "content_sentiment": -0.05
  }'
```

---

## Monitoring Stack

`docker-compose.monitoring.yml` defines:

- `stock-api` – The prediction API container (image: `noorulbaseer/stock-predictor:latest`, exposed on `8000`).
- `prometheus` – Configured via `prometheus.yml`, exposed on `9090`.
- `grafana` – Dashboard UI, exposed on `3000` (default admin password: `admin`).

Start monitoring stack:

```bash
cd "ETL Pipeline"
docker compose -f docker-compose.monitoring.yml up -d
```

Then visit:

- API: `http://localhost:8000` (FastAPI docs at `/docs`).
- Prometheus: `http://localhost:9090`.
- Grafana: `http://localhost:3000`.

---

## Triggering and Monitoring the DAG via Script

File: `scripts/trigger_and_monitor.py`

- Uses Airflow REST API (`/api/v2/dags/{dag_id}/dagRuns`) to trigger and then poll a run.
- Reads configuration from environment variables:
  - `ASTRO_AIRFLOW_URL` – Base URL of the Airflow instance.
  - `ASTRO_API_TOKEN` – Bearer token for authentication.
- Cleans/normalizes the Airflow URL, handles transient errors, redirects, and cold starts.
- Monitors run state for up to 20 minutes, failing if DAG run fails or times out.

Run (example):

```bash
setx ASTRO_AIRFLOW_URL "<your-airflow-url>"
setx ASTRO_API_TOKEN "<your-api-token>"
python scripts/trigger_and_monitor.py
```

(On PowerShell, prefer `$Env:ASTRO_AIRFLOW_URL = "..."` instead of `setx` for the current session.)

---

## Model Comparison & Champion Promotion

File: `scripts/compare_metrics.py`

- Connects to MLflow tracking server on DagsHub based on env vars:
  - `DAGSHUB_USERNAME`
  - `DAGSHUB_TOKEN`
  - `REPO_NAME`
- Looks up the latest run in experiment `Stock_Price_Prediction`.
- Compares its RMSE against the current champion model stored in MLflow Model Registry under name `Stock_Sentiment_Predictor` with alias `champion`.
- Generates a markdown report (`report.md`) summarizing candidate vs baseline.
- Optionally posts the report as a PR comment via `cml comment create` (if CML is installed).
- If the candidate is strictly better (lower RMSE), promotes it by assigning the `champion` alias to the candidate’s version.

---

## Setup & Installation

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development / scripts)
- Access to:
  - **GNews API key**.
  - **DagsHub** account + repository (for DVC/MLflow backends).
  - **GitHub** personal access token (for Airflow-based git operations).

### Python Environment

From the `ETL Pipeline` directory:

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
```

`requirements.txt` includes:

- `apache-airflow-providers-http`
- `dvc[s3]`
- `requests`, `pandas`, `textblob`, `pendulum`
- `ydata-profiling`, `mlflow`, `scikit-learn`, `joblib`

*(The inference image `noorulbaseer/stock-predictor:latest` already bundles its own runtime.)*

---

## Airflow Configuration

The DAG expects the following **Airflow Variables** to be set:

- `gnews_api_key` – GNews API key.
- `dagshub_access_key` – DagsHub access key (for DVC S3 remote).
- `dagshub_secret_key` – DagsHub secret key.
- `dagshub_username` – DagsHub username (for MLflow tracking URI).
- `dagshub_token` – DagsHub token (for MLflow auth).
- `github_token` – GitHub personal access token used by Git tasks.
- `github_username` – GitHub username.

Other important constants in the DAG:

- `REPO_NAME = "Real-Time-Predictive-System"` (DagsHub repo for MLflow/Bucket).
- `GITHUB_REPO = "Real-Time-Predictive-System-ETL-Pipeline"`.

Make sure these names match your actual DagsHub and GitHub repositories.

---

## CI/CD Workflows & Run Flow

This repository uses **GitHub Actions** to automate quality checks, remote retraining, and production image builds.

### Workflows

- **`CI - Feature to Dev`** (`.github/workflows/ci-feature.yml`)
   - Trigger: Pull Requests targeting the `dev` branch (feature → dev).
   - Steps:
      - Set up Python 3.9 and install dependencies with `uv`.
      - Run `flake8` for syntax and style checks.
      - Initialize an Airflow DB and run `pytest` against tests in `tests/`.
   - Purpose: Acts as a **quality gate** for feature branch before merging into `dev`.

- **`CI - Dev to Test (Remote Trigger)`** (`.github/workflows/ci-dev.yaml`)
   - Triggers:
      - Pull Requests targeting the `test` branch (dev → test).
      - Nightly schedule (`0 0 * * *`).
      - Manual `workflow_dispatch`.
   - Behavior:
      - Inspects the commit message:
         - If it contains `ETL Update`, treats it as a **bot/data** commit and skips heavy retraining, only checking status of the parent human commit.
         - Otherwise (human commit), runs the **full pipeline**:
            - Installs Astro CLI and wakes up the remote Astronomer deployment.
            - Deploys the latest DAG code to Astronomer Cloud.
            - Sets up Python + CML.
            - Runs `scripts/trigger_and_monitor.py` to trigger the `stock_news_pipeline` DAG remotely and wait for completion.
            - Runs `scripts/compare_metrics.py` to compare candidate vs champion in MLflow and (if better) promote the new model.
            - Hibernates the Astro deployment afterwards.
   - Purpose: Validates the **dev → test** promotion by actually retraining, logging metrics, and enforcing a model quality gate.

- **`CD - Production Deployment`** (`.github/workflows/ci-test.yaml`)
   - Triggers:
      - Pushes to `master` (test → master promotion).
      - Nightly schedule (`0 2 * * *`).
      - Manual `workflow_dispatch`.
   - Steps:
      - Uses Python to run `scripts/fetch_champion_model.py`, downloading the current MLflow champion from DagsHub.
      - Logs in to Docker Hub.
      - Builds an inference image from `inference/Dockerfile`, tagging it as `v1.0.<run_number>` and `latest`.
      - Runs a local container, hits the health endpoint at `http://localhost:8000/`, and stops the container.
      - Pushes both the versioned and `latest` tags to Docker Hub.
   - Purpose: Automates **packaging and publishing** the best model as a container image ready for deployment.

### End-to-End Run Flow (Branches → Production)

1. **Feature → Dev (Quality Gate)**  
    - Open a PR from a feature branch into `dev`.  
    - `CI - Feature to Dev` runs linting and tests; the PR should only be merged when this passes.

2. **Dev → Test (Remote Retrain & Evaluation)**  
    - Open a PR from `dev` into `test` or rely on the nightly schedule.  
    - `CI - Dev to Test (Remote Trigger)` deploys DAG code to Astronomer, triggers the Airflow DAG, and compares candidate vs champion metrics in MLflow.  
    - If the candidate is better, it is promoted to **champion** in the MLflow Model Registry.

3. **Test → Master (Container Build & Publish)**  
    - Merge from `test` into `master` (or wait for the nightly schedule).  
    - `CD - Production Deployment` fetches the champion model, builds the inference Docker image, health-checks it, and pushes it to Docker Hub as both a versioned tag and `latest`.

This flow connects **code changes → remote retraining → model selection → container build**, ensuring only vetted models reach production images.

---

## Running the Pipeline (Typical Flow)

1. **Start Airflow (Astro or local runtime).**
   - Ensure `dags/stock_news_pipeline.py` is visible to the scheduler.
   - Mount `data/`, `.dvc/`, and `.git/` according to `docker-compose.override.yml` if using containers.

2. **Configure Airflow variables and connections** (see above).

3. **Trigger the DAG**
   - From Airflow UI: Trigger `stock_news_pipeline` manually.
   - Or via script: `python scripts/trigger_and_monitor.py` (with proper `ASTRO_*` env vars).

4. **Inspect outputs**
   - Raw JSON in `data/raw/`.
   - Processed CSV in `data/processed/`.
   - Profiling report HTML in `data/reports/`.
   - MLflow runs and artifacts on DagsHub.

5. **Compare and promote models** (optional CI step)
   - Run `python scripts/compare_metrics.py` in a CI pipeline or locally to decide whether to promote the latest model.

6. **Serve the champion model**
   - Build and publish a prediction image that uses the champion model artifact.
   - Start `docker-compose.monitoring.yml` to run API + Prometheus + Grafana.

---

## Testing

There are basic DAG tests under `tests/` (e.g. `tests/test_dag.py`, plus Astro test utilities under `.astro/`). To run tests locally:

```bash
pytest
```

(You may need an Airflow-compatible environment and some Airflow variables mocked or set.)

---

## Notes & Limitations

- The target variable `market_change` is **synthetic** and exists for demonstration; in a real system you would join with actual market data.
- Some paths and repository names (DagsHub, GitHub) are **hard-coded**; update them if you fork/rename the project.
- Error handling is designed primarily for demo/teaching; you may want to harden retry policies, backoff, and secrets management for production.

---

## License

This project is for educational and demonstration purposes and is licensed under the **MIT License**. See the full text in the [LICENSE](LICENSE) file.
