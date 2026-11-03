# Fraud Detection V1: Power BI Dashboard

## Open it

Double-click `Fraud_Dashboard_V1.pbip` (needs Power BI Desktop).

If you see "Some of the tables have incomplete or no data", click **Refresh now**.

## Pages

| Page | What it shows |
|---|---|
| Fraud Overview | Headline numbers, fraud by type, amount and hour period, with a type filter |
| Model Performance | Final model on the test set, model selection, errors, feature importance |
| Validation Checks | What happens when data is restricted, including the no-balance check |
| Transaction Explorer | Filter and inspect individual transactions and model outcomes |

## Make a single .pbix file

In Power BI: **File > Save as**, then choose **Power BI file (*.pbix)**.

## If you move the folder

The data path is stored in a parameter. In Power BI: **Transform data > Edit parameters**, then set
`DataFolder` to the new `dashboard\data\` path (keep the last backslash) and click **Refresh**.

## Rebuild the data

```bash
python prepare_dashboard_data.py
```

## Notes

- All figures describe a simulated PaySim sample, not real customers.
- Model scores use threshold 0.5. Scores on Train rows are optimistic, so use Split = Test to judge the model.
