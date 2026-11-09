# Pharmacy Medicine Demand & Expiry-Risk Forecasting

A local, reproducible AI/ML internship project that forecasts weekly medicine demand and estimates short-dated inventory exposure with a first-expiry-first-out (FEFO) simulation.

[Open the notebook in Google Colab](https://colab.research.google.com/github/nithinkumarreddy45/Assignments/blob/main/pharmacy-medicine-demand-expiry-risk-forecasting/Pharmacy_Medicine_Demand_Expiry_Risk_Forecasting.ipynb)

## What is included

- 24 fictional medicine SKUs across four categories and 104 weeks of synthetic demand
- A per-SKU ridge regression model using trend, annual seasonality, and demand lags (1, 4, 13, and 52 weeks)
- A 12-week recursive demand forecast with approximate residual-based ranges
- A chronological 12-week holdout evaluation against a recent four-week-average baseline
- A lot-level expiry-risk simulation and a simple lead-time replenishment estimate
- A trained model artifact, CSV evidence, and an exploratory notebook

All records are generated locally with a fixed random seed. The project contains no real pharmacy, patient, or student data and makes no external API calls.

## Quick start

Use Python 3.10+ in a local environment. Install the small set of dependencies and run:

```bash
python -m pip install -r requirements.txt
python run_project.py
```

The run regenerates the synthetic CSV files, trains and saves the model, and refreshes outputs in `evidence/` and `models/`.

To open the notebook, start Jupyter from the project folder:

```bash
python -m pip install jupyter
jupyter notebook Pharmacy_Medicine_Demand_Expiry_Risk_Forecasting.ipynb
```

## Project layout

- `data/weekly_demand.csv` - synthetic weekly sales history and a context-only promotion flag
- `data/products.csv` - fictional SKU categories, unit costs, and lead times
- `data/inventory_lots.csv` - fictional on-hand lot quantities and expiry dates
- `src/generate_data.py` - seeded synthetic data generator
- `src/forecast_model.py` - NumPy ridge regression and chronological evaluation helpers
- `src/pipeline.py` - forecasting, evaluation, FEFO simulation, and reorder estimates
- `models/demand_model.json` - fitted per-SKU parameters and model metadata
- `evidence/12_week_demand_forecast.csv` - point forecasts and approximate ranges by SKU and week
- `evidence/holdout_metrics_by_sku.csv` - per-SKU holdout metrics and baseline comparison
- `evidence/evaluation_summary.json` - aggregate holdout summary
- `evidence/expiry_risk_and_reorder.csv` - expiring units, estimated cost exposure, and suggested order quantities

## Method

For evaluation, the final 12 weeks of each SKU history are held out. The model is fit on earlier weeks and predicts the holdout recursively, so each predicted week feeds the next forecast. The same holdout is compared with a baseline that repeats the mean of the last four training weeks. The saved production-style demo model is then refit on all 104 historical weeks and forecasts the next 12 weeks.

On the generated data, aggregate holdout WAPE is **13.48%** for the model and **13.75%** for the baseline, about a **1.94% relative reduction** in WAPE. This is a small gain on one synthetic sample and does not establish real-world performance. The per-SKU results in the CSV are important because the aggregate can hide weak categories or individual products.

For expiry exposure, the pipeline consumes projected demand from the earliest-expiring usable lot first. Units still present when their expiry week arrives are counted as at-risk. The risk value multiplies those units by the fictional unit cost. The suggested order quantity uses forecast demand over the product lead time plus a residual-based safety buffer, less current on-hand stock.

## Important limitations

- This is an educational prototype, not a procurement, clinical, or regulatory system.
- The synthetic history has no true stockouts, substitutions, supplier delays, changing prices, recalls, or censored demand. The promotion column is retained as context and is not used as a model feature.
- The approximate forecast ranges use one-step residual spread; they are not calibrated multi-step prediction intervals.
- Expiry estimates depend on the accuracy of the lot snapshot and forecast and assume FEFO consumption with no replenishment during the 12-week horizon.
- Before any real-world use, replace the synthetic data with appropriately authorized, de-identified data, validate across chronological backtests and product groups, check data quality, and obtain pharmacy operations and compliance review. Do not use identifiable patient information.

## AI use disclosure

AI assistance was used to help draft project code, documentation, and analysis. The implementation and outputs should be reviewed and understood by the submitting student before submission. Generated data and all model outputs are synthetic and reproducible from the included code.

