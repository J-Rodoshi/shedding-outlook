# Shedding Outlook — Bangladesh National Load-Shedding Forecasting

**Beyond Persistence: When Do System, Weather and Calendar Data Improve Prediction of National Load Shedding in Bangladesh?**

[![CI](https://github.com/J-Rodoshi/shedding-outlook/actions/workflows/ci.yml/badge.svg)](https://github.com/J-Rodoshi/shedding-outlook/actions/workflows/ci.yml)
[![Docker image publishing](https://github.com/J-Rodoshi/shedding-outlook/actions/workflows/publish.yml/badge.svg)](https://github.com/J-Rodoshi/shedding-outlook/actions/workflows/publish.yml)

**MSc Data Science and Artificial Intelligence — Asian Institute of Technology (2026)**  
**Authors:** Jarin Tasnim Rodoshi and Khaing Mon Kyaw  
**Research notebook:** `Load_Shedding_Final_Project.ipynb` (maintained in the original research project; not required to run this app)

Shedding Outlook is a research-backed web application for exploring **national, operator-reported load-shedding occurrence in Bangladesh** at **1-hour, 6-hour, and 24-hour horizons**. It packages historical, **out-of-sample predictions** and separately trained XGBoost models behind a responsive FastAPI application.

The research question is not simply *“Can machine learning predict shedding?”* It is: **Does additional electricity-system, weather, and calendar information improve future forecasts over a strong baseline that already knows the current shedding state?**

> **Scope:** This project predicts a **national binary indicator** of reported shedding. It does **not** identify outages on particular streets, homes, districts, or divisions. The public interface browses **historical evaluated forecasts**, not an automatically updated live forecast. A separate API performs inference when a complete engineered input vector is supplied.

## Project highlights

| Item | Implementation |
| --- | --- |
| Research inputs | PGCB hourly electricity-system records; nationally aggregated ERA5 weather; Bangladesh calendar and holidays |
| Modelling window | 2022–March 2026 |
| Modelling-ready origin records | 29,448 |
| Target | `y(t, h) = 1` when reported national shedding at `t + h` is positive; `h ∈ {1, 6, 24}` hours |
| Principal comparison | **12-feature state-aware XGBoost** vs **28-feature expanded XGBoost** |
| Evaluation | Nine expanding-window, quarterly out-of-time folds, January 2024–March 2026 |
| Ensemble | Seeds 0, 1 and 2; separate models per horizon |
| Historical archive | 16,099 distinct evaluated forecast-origin hours; horizon-specific scored samples differ |
| Model serving artifacts | 18 final-refit XGBoost models: 2 configurations × 3 horizons × 3 seeds |
| Web stack | FastAPI, Uvicorn, HTML/CSS/JavaScript |
| Engineering | pytest, Docker, GitHub Actions CI, GHCR, AWS EC2 |

## Model architecture and forecasting pipeline

The figure separates **historical rolling-origin evaluation** from **final-refit model serving**. Those are related but distinct workflows.

![Shedding Outlook technical model architecture: inputs, preprocessing, origin-time features, two XGBoost branches, three horizons, and separate evaluation and model-serving artifacts](docs/model_pipeline.svg)

[Open the full-size model architecture (SVG)](docs/model_pipeline.svg)

### 1. Data acquisition and preprocessing

Three complementary sources are integrated:

1. **PGCB electricity data:** Reported national shedding, demand, generation, fuel mix and cross-border import flows. The original research source contains **94,934 raw observations**.
2. **ERA5 weather data:** Temperature, precipitation, humidity, wind speed and solar radiation, aggregated over **52 grid points** inside Bangladesh. Weather in the core research analysis is retrospective reanalysis.
3. **Bangladesh calendar:** Local time features, public holidays and the **Friday–Saturday weekend**, including checks on relevant observed holidays.

The notebook audits suspicious measurements, aligns the sources to Bangladesh local hourly timestamps, handles missing observations, and constructs the model-ready table. The modelling sample contains **29,448 origin rows**.

### 2. Leakage-aware targets and features

For each forecast-origin hour `t` and lead `h`:

```text
y(t, h) = 1 if reported national load shedding at t + h > 0 MW
          0 otherwise
h = 1, 6, or 24 hours
```

The main features are constructed at the **forecast origin**, not from the realized future target. They include:

- **Persistence and state:** Current shedding status, current demand–generation gap, lagged shedding (1, 24 and 168 hours), and rolling 24/168-hour summaries.
- **System conditions:** Demand, generation, lagged system measurements and selected India electricity-import series.
- **Weather:** National temperature, precipitation, humidity, wind speed and solar radiation.
- **Calendar:** Hour, cyclical hour encoding, day of week, weekend and holiday indicators.

The horizon labels are joined at their **exact future timestamps** to avoid silently shifting missing observations to the wrong future hour.

### 3. Controlled model comparison

Both principal configurations use the **same XGBoost algorithm and fixed hyperparameters**. The scientific comparison is about the value of the added **information**, not about switching to a different learning algorithm.

| Configuration | Features | Question answered |
| --- | ---: | --- |
| **State-aware baseline** (`pers+state+hour`) | 12 | What is predictable from recent shedding, current shedding state and time? |
| **Expanded XGBoost** (`frozen (exploratory)`) | 28 | Do system, weather and calendar variables add value beyond the strong baseline? |

The core classifier uses **300 trees**, **maximum depth 6**, **learning rate 0.05**, and **0.8** for both row and feature subsampling. Predictions from seeds **0, 1 and 2** are averaged. Separate classifiers are fitted for the 1-, 6- and 24-hour targets.

Additional research comparisons include carry-forward and seasonal rules, state-by-hour lookup, logistic regression, histogram gradient boosting, and a **training-only drift-screen** feature-selection alternative. The fixed 28-feature subset is **exploratory**, because it was finalized after inspecting evaluation folds.

### 4. Chronological evaluation

The principal experiment uses **nine quarterly test folds** from 2024 Q1 through 2026 Q1. Each fold fits on earlier data and scores a subsequent quarter. A **horizon-dependent gap** separates training from evaluation to protect the forecast target boundary.

- **Primary metric:** Brier score — lower is better for probability prediction.
- **Additional metrics:** ROC-AUC, PR-AUC, calibration diagnostics, transition analysis and quarterly robustness.
- **Uncertainty:** Paired moving-block bootstrap intervals for model-minus-baseline differences.
- **Horizon-specific scored origins:** **16,099 (1 h)**, **15,307 (6 h)** and **16,059 (24 h)**. Do not assume every archived hour has all three valid observed outcomes.

### 5. Results from the final research report

| Forecast horizon | State-aware Brier ↓ | Expanded Brier ↓ | Relative reduction | Interpretation |
| --- | ---: | ---: | ---: | --- |
| **1 hour** | 0.0561 | 0.0542 | **3.4%** | Small improvement |
| **6 hours** | 0.1400 | 0.1230 | **12.1%** | Clearest supported improvement |
| **24 hours** | 0.1484 | 0.1436 | **~3.2%** | Difference not established: confidence interval includes zero |

At **6 hours**, the model-minus-baseline Brier difference is **−0.0170** (95% interval **−0.0248 to −0.0095**). Gains are especially important around retrospectively identified **shedding onsets and stops**. However, the expanded model performs worse in **January–March 2025**, a period coinciding with a substantial change in India–Tripura electricity imports. Feature-ablation results support a **distribution-shift hypothesis** but do not prove a causal explanation.

Calibration also matters: approximate pooled slopes are **0.97 (1 h), 0.72 (6 h), and 0.67 (24 h)** for the expanded model. Longer-horizon outputs are better regarded as **risk rankings** than precise operational probabilities.

**Exploratory extension:** On a matched subset of **15,692 origins**, adding target-hour calendar features and archived day-ahead forecast weather reduced 24-hour Brier from **0.1464 to 0.1331**. This was developed after inspection of earlier results and is **not an independent deployment validation**.

> **Reproducibility note:** The displayed headline values are from the final research report. Re-exported serving predictions may have small numerical differences from the original notebook due to dependency and numeric-processing differences; the exporter validates Brier against the saved research summary to within 0.001.

## How the web application works

Shedding Outlook presents the research findings through three tabs:

- **Today:** Select a **historical date, hour and display location**. View the archived national shedding state, the 1/6/24-hour forecast levels, and the national outcomes subsequently observed in the data.
- **Typical:** Explore descriptive shedding frequencies by year, holiday status and national-temperature decile. These are descriptive associations, not causal estimates.
- **Analyst:** Compare Brier scores, inspect group-level feature importance and examine the quarterly record, including the failed 2025 Q1 evaluation.

**Location selector:** Bangladesh and all eight divisions are listed for navigation, but **selecting a division does not change the national forecast**, because the dataset has no division-specific prediction target.

**Risk labels:** `Low` (< 0.25), `Elevated` (0.25 to < 0.60), and `High` (≥ 0.60) are **interface thresholds**, not validated public warning levels.

There are **two distinct inference modes**:

| Mode | API | What it returns |
| --- | --- | --- |
| **Historical, out-of-sample lookup** | `GET /api/hour` | Predictions made by fold-specific models for evaluated historical origin hours, alongside available observed outcomes |
| **Final-refit model inference** | `POST /api/predict-from-features` | Predictions generated now by saved models, **only when all required engineered features are supplied** |

The website does **not** connect to a real-time PGCB feed and does **not** automatically construct inputs for today's actual grid conditions. The application intentionally does not fabricate predictions for dates outside its evaluated archive.

## Repository layout

```text
shedding-outlook/
├── app/
│   └── main.py                         # FastAPI app and endpoints
├── web/
│   ├── index.html                      # Responsive web page
│   ├── static/app.js                   # Interactive behaviour and charts
│   └── data/
│       ├── outlook.json                # Archived out-of-sample predictions
│       ├── report.json                 # Research metrics and EDA summaries
│       └── export_validation.csv       # Export comparison checks
├── artifacts/
│   ├── models/                         # 18 XGBoost JSON models + manifest
│   └── source_provenance.json          # Data provenance and validation metadata
├── docs/
│   └── model_pipeline.svg              # Scientific architecture diagram
├── scripts/
│   ├── export_outlook.py               # Recompute archived forecasts and refit models
│   ├── build_report_json.py            # Build descriptive/analyst summaries
│   └── check_artifacts.py              # Confirm real, non-synthetic artifacts
├── tests/test_api.py                   # API and artifact-integrity tests
├── .github/workflows/
│   ├── ci.yml                          # CI on push / PR
│   ├── publish.yml                     # Manual Docker image publishing
│   └── deploy.yml.disabled             # Optional EC2 auto-deploy, NOT enabled
├── Dockerfile
├── requirements-api.txt
├── requirements-dev.txt
├── requirements-train.txt
└── README.md
```

This repository includes the **application and pre-exported serving artifacts**, but not the complete research notebook or raw Parquet/PGCB data. **You can run the app without retraining.**

## Run locally

Requires **Python 3.11**. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS:       source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

| Service | Local address |
| --- | --- |
| Application | http://127.0.0.1:8000 |
| Health check | http://127.0.0.1:8000/health |
| Interactive API documentation | http://127.0.0.1:8000/docs |

Example historical lookup (date is within the archived range):

```http
GET /api/hour?date=2025-07-15&hour=10&location=bangladesh
```

For final-refit inference, inspect `artifacts/models/manifest.json` for the exact **28 expanded-model features** and required names. Supply them as a JSON object:

```json
{
  "origin": "2026-03-07T10:00",
  "features": {
    "hour": 10
  }
}
```

This intentionally **incomplete** example returns HTTP 422 with the names of missing features. It does **not** generate a prediction from one feature. Provide all required, correctly engineered values to run the saved model ensemble.

## Run with Docker

Build and run from the repository root:

```bash
docker build -t shedding-outlook:local .
docker run -d --name shedding-outlook-local --restart unless-stopped -p 8002:8000 shedding-outlook:local
```

Then visit http://127.0.0.1:8002 or check:

```bash
curl -fsS http://127.0.0.1:8002/health
docker logs shedding-outlook-local
```

Remove the local container when finished:

```bash
docker rm -f shedding-outlook-local
```

The image uses `python:3.11-slim`, runs Uvicorn under a non-root user with **one worker**, and exposes port **8000** internally. The archived JSON and refit models are bundled into the image; notebook retraining and raw source data are **not required** on the server.

## CI, Docker image publishing and AWS deployment

The implemented release process is:

```text
GitHub source code
    └── GitHub Actions CI (automatically on push / PR)
          ├── Python 3.11 dependency installation
          ├── pytest suite (13 tests)
          └── Docker image build verification

GitHub Actions "Publish Docker image to GHCR" (manual workflow_dispatch)
    └── Build and publish ghcr.io/j-rodoshi/shedding-outlook:latest
          └── Pull container image to AWS EC2 manually
                └── Docker: public HTTP 80 → container HTTP 8000
                      └── FastAPI + web interface + archived forecasts + models
```

- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs tests and builds a container for validation.
- **Image publishing:** [`.github/workflows/publish.yml`](.github/workflows/publish.yml) is manually triggered from the GitHub Actions tab, and tags each image with `latest` and the source commit SHA.
- **EC2 deployment:** The published GHCR image was deployed to an **Amazon Linux 2023 `t3.small`** instance (**2 vCPUs, 2 GiB RAM, 20 GiB gp3 EBS**). The application was exposed on HTTP port 80, and the container used a restart policy.
- **Current automation boundary:** **Continuous integration is enabled, but automatic GitHub-to-EC2 redeployment is not.** The `deploy.yml.disabled` file is retained as an optional template; it is **not an active GitHub Actions workflow**.

### Manual EC2 deployment (Amazon Linux 2023)

Create a suitable EC2 instance, a restricted SSH key pair, and security-group rules permitting SSH (preferably only from your current IP) plus HTTP port 80 for the demo. **Never publish a private SSH key to GitHub.**

SSH into the instance and install Docker:

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# Reconnect to activate Docker group membership.
docker run --rm hello-world
```

After the GitHub image publishing workflow succeeds and the GHCR package is **public**, pull and run it on EC2:

```bash
docker pull ghcr.io/j-rodoshi/shedding-outlook:latest
docker rm -f shedding-outlook 2>/dev/null || true
docker run -d \
  --name shedding-outlook \
  --restart unless-stopped \
  --memory 1500m \
  -p 80:8000 \
  ghcr.io/j-rodoshi/shedding-outlook:latest
curl -fsS http://127.0.0.1/health
docker ps
```

If the GHCR package is private, authenticate Docker using appropriately scoped credentials before pulling. An EC2 public IP may change when the instance is stopped and restarted. **HTTP is not encrypted**; production use would require TLS/HTTPS and appropriate access controls.

### Computational footprint

In the final research report's **single-CPU last-fold experiment** (not an AWS benchmark):

| XGBoost configuration | Fit time | Prediction time per 1,000 rows |
| --- | ---: | ---: |
| 12-feature baseline | 0.31 s | 3.64 ms |
| 28-feature expanded | 0.53 s | 4.30 ms |

These figures measure **model-only computation on the reported experimental workload**. They do **not** measure EC2 website latency, network time, cloud charges or uptime.

## Regenerate model and prediction artifacts

The already-committed artifacts are sufficient to run the web application. To rerun the source-based exporter, place the notebook's original outputs in an **external data directory**, including:

```text
modeling_ready.parquet
checkpoint_7_regularized.parquet
results/final_summary.csv
results/fold_summary.csv
results/grouped_permutation_importance.csv
```

From the app repository root:

```bash
python -m pip install -r requirements-train.txt
python scripts/export_outlook.py --data "/path/to/CP Electricity Project"
python scripts/build_report_json.py --data "/path/to/CP Electricity Project"
python scripts/check_artifacts.py
python -m pytest -q tests/
```

The exporter fits fold-specific models to reconstruct the historical prediction archive and, separately, saves the **18 final-refit serving models**. It checks numerical agreement with the research summary when that file is present. The export script **does not call an external real-time weather or PGCB API**, and fails if required data or features are missing.

## Research interpretation and responsible use

1. **National scope only:** A national shedding event is **not** a specific household, street or division outage.
2. **Historical web forecasts:** The archived website observations are out-of-sample *historical* predictions through March 2026. They are not forecasts for today's grid conditions.
3. **Live inference requirements:** A caller can run the final-refit models only after obtaining timely, valid measurements and reproducing the original feature engineering. There is no automatic live ingestion pipeline here.
4. **Feature-selection caution:** The fixed 28-feature model was selected after looking at evaluated quarters; the main result is **exploratory**, with supporting training-only selection checks.
5. **Temporal robustness:** The 2025 Q1 import-regime shift and weaker long-horizon calibration are important findings, not failures to hide.
6. **Prediction meaning:** The target is whether the reported national shedding deficit is **positive** at the selected future hour—not the magnitude, location or duration of an outage.
7. **Risk labels:** `Low / Elevated / High` are user-interface categories, not validated operational alarms.

## Data sources, references and attribution

- [Hourly Electricity Generation, Demand, Load Shedding, and Fuel Mix Dataset for Bangladesh (2015–2026)](https://data.mendeley.com/datasets/vpk8spw2mm/1), Mendeley Data, DOI: [10.17632/vpk8spw2mm.1](https://doi.org/10.17632/vpk8spw2mm.1); based on PGCB-reported electricity-system records.
- Hersbach et al. (2020), *The ERA5 global reanalysis*, **Quarterly Journal of the Royal Meteorological Society**.
- Muñoz-Sabater et al. (2021), *ERA5-Land: A state-of-the-art global reanalysis dataset for land applications*, **Earth System Science Data**.
- [GADM](https://gadm.org/) boundary data for the national weather grid mask.
- [Python holidays](https://pypi.org/project/holidays/) for calendar feature generation.
- [Open-Meteo historical weather documentation](https://open-meteo.com/en/docs/historical-weather-api), describing the historical source access used by the research notebook. **The deployed app does not call Open-Meteo.**

**License and redistribution:** See [`LICENSE.txt`](LICENSE.txt). Code and model artifacts are provided for educational use; the original data providers retain their attribution and licensing requirements. The forecasts are research outputs, **not guaranteed public outage alerts**.
