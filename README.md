# Shedding Outlook — Bangladesh

**One project, two research-backed prediction modes:**

1. **Historical evaluation (the default website):** **real out-of-sample** predictions at 1, 6 and 24 hours from the 12-feature persistence baseline and exploratory 28-feature XGBoost model. Nine rolling quarterly test periods (January 2024–March 2026), seeds 0–2 and the horizon-dependent training gap match the final notebook.
2. **On-demand inference:** the FastAPI `/api/predict-from-features` endpoint accepts a complete feature vector and runs the **final-refit versions** of these same model families. It does not obtain current measurements automatically.

The website preserves the beginner-friendly **Today / Typical / Analyst** layout. The date and hour selector reads out-of-sample results for the chosen historic origin. The location selector shows Bangladesh and its eight divisions but **does not change the national-level probability**. There is no district-level ground truth in the study. The project has **no Open-Meteo or other external data API**.

**Coverage:** 16,099 evaluated origin hours, 2024-01-01 02:00 through 2026-03-08 22:00 (Asia/Dhaka). The exported models used PGCB + national weather + calendar inputs already engineered in the notebook.

## Contents

```text
shedding-outlook/
  app/main.py                 FastAPI, historical API, final-refit inference API
  web/index.html              Responsive 3-tab interface
  web/static/app.js           Date / time / location controls, charts
  web/data/outlook.json       Actual out-of-sample predictions and observed outcomes
  web/data/report.json        Original paper metrics and EDA
  web/data/export_validation.csv     Export scores
  artifacts/models/           18 final-refit XGBoost JSON models + manifest
  artifacts/source_provenance.json   Source hashes, validation and limitations
  scripts/export_outlook.py   Rebuild all archived predictions + models
  scripts/build_report_json.py   Rebuild report metrics from the notebook CSVs
  scripts/check_artifacts.py  Verify not synthetic
  tests/test_api.py           API and data-integrity tests
  Dockerfile                  AWS EC2 Docker image
  .github/workflows/ci.yml    Unit tests and Docker build
  .github/workflows/deploy.yml  Publish GHCR, deploy exact commit to EC2
```

The raw notebook files and Parquet datasets are **not included** in the GitHub/Docker build. The exported app can run immediately from the bundled artifacts.

## 1. Run locally (Windows PowerShell)

Unzip the project into, for example:

`C:\Users\User\Desktop\CP 2026\CP Electricity Project\GridLens_BD_App_and_Deployment\shedding-outlook`

From inside that **shedding-outlook** directory:

```powershell
$pythonExe = "C:\Users\User\Downloads\class assignment 4\.venv-gpu\Scripts\python.exe"
& $pythonExe -m pip install -r requirements-dev.txt
& $pythonExe -m pytest -q tests/
& $pythonExe -m uvicorn app.main:app --app-dir . --host 127.0.0.1 --port 8000
```

Open:

- Website: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Historical forecast lookup: `GET /api/hour?date=2025-07-15&hour=10&location=dhaka`

**The default date is the last origin with all three prediction horizons available.** Later origin hours near the data cutoff have fewer valid future outcomes, so those are shown as unavailable rather than invented.

The location dropdown does not promise division-level risk. All outputs are national-level estimates.

### Model serving API

Use FastAPI's interactive `/docs` page to try `POST /api/predict-from-features` with a JSON input:

```json
{
  "origin": "2026-03-07T10:00",
  "features": {
    "hour": 10
  }
}
```

**The sample above is deliberately incomplete:** the endpoint responds with a list of the missing input columns, rather than generating a bogus prediction. Provide all 28 engineered features with the exact meanings and units from the notebook to obtain actual final-refit inference. A proper live integration must obtain timely PGCB readings, feed those through the *same* origin-time feature engineering, and then call this endpoint. There is no operational live-data source in this repository.

## 2. Validate Docker on your laptop

With Docker Desktop open, from the project root:

```powershell
docker build -t shedding-outlook:local .
docker run -d --name shedding-outlook-local -p 8002:8000 shedding-outlook:local
Invoke-RestMethod http://127.0.0.1:8002/health
docker logs shedding-outlook-local
docker rm -f shedding-outlook-local
```

Do not confuse the Docker host port `8002` with the container port `8000`. You may select another unused host port.

## 3. Push to a new GitHub repository

Create a GitHub repository named `shedding-outlook`, without initializing a README. Then, from the **shedding-outlook** folder:

```powershell
git init -b main
git add .
git status
git commit -m "Deploy research-backed national shedding outlook"
git remote add origin https://github.com/YOUR_USERNAME/shedding-outlook.git
git push -u origin main
```

Only the serving artifacts, source code and supporting summary CSVs belong in the application repository. Do **not** commit `.venv`, `.pem`, AWS credentials, raw PGCB source CSVs, private API keys or other local project folders. Check `git status` before the commit.

Note: the `deploy.yml` workflow triggers on pushes to `main`. **Configure your EC2 secrets before the first production push**, or expect its deploy job to fail until configured; local CI and GHCR publishing can still succeed.

## 4. Manual first deployment on AWS EC2

This is the same deployment pattern as your Bangkok Taxi API project.

1. Start your AWS Learner Lab or AWS account. Launch an **Amazon Linux 2023** EC2 instance. Select an instance with sufficient memory for the XGBoost runtime (4 GiB is preferable; the app's memory use must be checked on your selected instance). Choose a key pair and keep the downloaded `.pem` outside GitHub.
2. Set security group inbound rules: **TCP 22** from **your public IP only**, **TCP 80** from the audience who will access the demo. HTTPS (443) needs a separate TLS/proxy setup; allowing 443 alone does not create HTTPS.
3. Record the public IP or associate an Elastic IP as allowed by your account; public IPs and Learner Lab availability can change. SSH from PowerShell:

```powershell
$key = "$HOME\Downloads\your-ec2-key.pem"
$ip = "YOUR_EC2_PUBLIC_IP"
ssh -i $key ec2-user@$ip
```

4. On EC2, install Docker:

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
exit
```

Reconnect to activate the Docker group, then verify `docker info`.

5. After pushing to GitHub, clone your repo on EC2, build and run:

```bash
git clone https://github.com/YOUR_USERNAME/shedding-outlook.git
cd shedding-outlook
docker build -t shedding-outlook:latest .
docker run -d --name shedding-outlook -p 80:8000 \
  --restart unless-stopped --memory 2g shedding-outlook:latest
curl -fsS http://127.0.0.1/health
```

The site should then be available at `http://YOUR_EC2_PUBLIC_IP/` if the security group and instance routing permit it.

Private GitHub repositories require appropriate clone credentials; a public repository avoids that particular hurdle.

## 5. Enable GitHub Actions deployment

The repository contains two workflows:

- `ci.yml`: install dependencies, test and build Docker.
- `deploy.yml`: test, publish an image to **GitHub Container Registry (GHCR)** tagged with the exact commit SHA, then SSH into EC2, pull and replace the live Docker container.

In GitHub → repository → **Settings → Secrets and variables → Actions**, add:

- `EC2_HOST`: your EC2 public IP or DNS.
- `EC2_SSH_KEY`: the full matching private key file, including header/footer. Store **only** as a GitHub secret.
- `EC2_KNOWN_HOSTS`: the **verified SSH host public-key entry** for your EC2 server (not a guessed value). Retrieve and verify against the host's trusted fingerprint. This enables strict host-key checking in CI.

For a **private** GHCR package, authenticate Docker to GHCR on EC2 with a GitHub token having `read:packages` and access to that package, before enabling automated pulls. Store credentials on the instance securely; never commit them. Alternatively make the package public and allow unauthenticated pulls.

After configuring the secrets and GHCR visibility/permissions, run the **Publish and deploy to AWS EC2** workflow from the Actions tab. Confirm the workflow runs green, then verify `/health` and the webpage on the EC2 IP. Subsequent pushes to `main` also trigger the workflow.

AWS charges may apply. Learner Lab EC2 instances and public IPs may be short-lived. No AWS/GitHub accounts are modified by the ZIP itself.

## 6. Retrain/re-export using *your notebook* data

The package already contains finished models and historical predictions. You do not need to retrain to start the app.

If you modify the original notebook or want to regenerate the serving assets, make sure your project folder contains:

- `modeling_ready.parquet`
- `checkpoint_7_regularized.parquet`
- `results/final_summary.csv`
- `results/fold_summary.csv`
- `results/grouped_permutation_importance.csv`

Then run from inside `shedding-outlook`:

```powershell
& $pythonExe -m pip install -r requirements-train.txt
& $pythonExe scripts/export_outlook.py `
  --data "C:\Users\User\Desktop\CP 2026\CP Electricity Project" `
  --out "web/data/outlook.json" --models "artifacts/models"
& $pythonExe scripts/build_report_json.py `
  --data "C:\Users\User\Desktop\CP 2026\CP Electricity Project" `
  --out "web/data/report.json"
& $pythonExe scripts/check_artifacts.py
& $pythonExe -m pytest -q tests/
```

The exporter trains 12-/28-feature XGBoost models with seeds 0, 1, 2 for each eligible quarter and horizon, creates out-of-sample forecasts, validates Brier scores against your `final_summary.csv` (within 0.001 to allow small package-version differences), and separately saves 18 final-refit models for API inference.

There is **no synthetic fallback**. If required files/features are missing or validation fails, the exporter stops with an error. The original report’s headline results are preserved unchanged in the Analyst section.

## 7. Scientific caveats for your demonstration

- The study's target is **national operator-reported load shedding**, not household or neighbourhood outages.
- Only the 2024–2026 historical quarter predictions have an out-of-sample performance interpretation. The final-refit API models are for serving supplied feature vectors and **were not directly scored using the same untouched test periods**.
- The 28-feature configuration is **exploratory**; it was selected after examining evaluated periods, so the apparent gain can be optimistic. The strongest supported gain is around six hours.
- The saved prediction scores generated in this Python environment match the notebook summary within 0.001 Brier; small numerical differences can arise from dependency versions or numeric representations.
- The `Low / Elevated / High` boundaries (`<0.25`, `0.25–<0.60`, `>=0.60`) are **presentation thresholds**, not scientifically validated action categories. Longer-horizon raw probabilities are miscalibrated and should not be treated as precise individual risk guarantees.
- The source data end **8 March 2026**. There is **no direct feed** for 23 November 2026 or later. Historical date selection is factual; new-date predictions require current, correctly engineered 28-feature inputs.
- Neither a new calendar-only model nor a new Open-Meteo dependency is used.
