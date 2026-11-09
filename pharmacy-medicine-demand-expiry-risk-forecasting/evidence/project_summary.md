# Run summary

- Snapshot date: 2026-10-26 (synthetic)
- Products: 24 fictional SKUs
- History: 104 weekly observations per SKU
- Evaluation: rolling recursive forecast over the final 12 holdout weeks
- Model WAPE: 13.48%
- Four-week-average baseline WAPE: 13.75%
- Relative WAPE reduction: 1.94%
- Future forecast: 12 weeks for each SKU

The model improvement is modest and is specific to this generated sample. Review `holdout_metrics_by_sku.csv` for SKU-level variation. `expiry_risk_and_reorder.csv` contains synthetic FEFO expiry exposure and indicative replenishment quantities; it does not place or authorize orders.

