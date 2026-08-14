# Ames Housing Price Prediction

This project trains a house-price prediction model and exposes it through a
FastAPI service. The API accepts raw Ames Housing features and returns the
predicted sale price in US dollars.

## Requirements

- Python 3.11 (recommended)
- PowerShell on Windows

## Setup

Create and activate a virtual environment from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current terminal and
then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter so
that imports resolve correctly.

## Run the tests

```powershell
pytest -q
```

The expected result is a passing test suite. A `StarletteDeprecationWarning`
from FastAPI's test client is a dependency warning and does not indicate an
application error.

## Start the API

The trained artifacts must be present at `models/model.pkl` and
`models/features.pkl`.

```powershell
uvicorn api.main:app --reload
```

Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

Available endpoints:

- `GET /health` reports whether the model artifacts are ready.
- `POST /predict` validates one house payload and returns a predicted sale
  price.

`/predict` requires every raw feature listed in `models/features.pkl` under
`raw_columns`. The Swagger page at `/docs` shows the generated request schema.

## Optional Databricks logging

After a successful prediction, the API can store the input features,
prediction, timestamp, and model version in a Databricks Delta table.

Copy `.env.example` to `.env`, then replace its placeholder values:

```powershell
Copy-Item .env.example .env
```

Configure these values in `.env`:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_WAREHOUSE_ID`
- `DATABRICKS_PREDICTIONS_TABLE`

In Databricks, open a SQL editor and run
[`databricks/create_predictions_table.sql`](databricks/create_predictions_table.sql).
If you choose a different catalog, schema, or table name, use that exact fully
qualified name for `DATABRICKS_PREDICTIONS_TABLE` in `.env`.

Restart the API after changing `.env`, make one successful `POST /predict`
request from `/docs`, then run the verification query included in the SQL file.
You should see the input JSON, predicted price, UTC timestamp, and model hash.

The `.env` file is ignored by Git and must not be committed. If the values are
not configured, predictions still work; only Delta Lake logging is skipped.

## Start the Streamlit dashboard

The dashboard calls the FastAPI service for every prediction. Start the API
first, then launch Streamlit from the project root:

```powershell
uvicorn api.main:app --reload
streamlit run streamlit/app.py
```

The dashboard provides:

- **Single prediction** — edit the five main house features, optionally expand
  advanced details, and submit one house to `POST /predict`.
- **Batch prediction** — upload a CSV with the raw model columns and download
  the results with a `PredictedSalePrice` column.

Form defaults come from [`models/defaults.json`](models/defaults.json), which
P4 prepared as the baseline house profile. Override the API location with
`STREAMLIT_API_URL` in `.env` when the service is not running locally.
