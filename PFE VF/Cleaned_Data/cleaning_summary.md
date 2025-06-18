# Data Cleaning Summary

**Date:** 2025-06-17 04:51:30
**User:** AmineZouaghi

## Dataset Overview

* Original size: 92943 rows × 21 columns
* Cleaned size: 92943 rows × 21 columns

## Actions Performed

1. Removed 0 duplicate rows
2. Fixed 18066 missing values
3. Corrected 29 negative values
4. Capped 91820 outliers using IQR method
5. Standardized binary columns to contain only 0 and 1

### Missing Value Treatment Details

| Column | Missing Count | Method |
|--------|--------------|--------|
| Min_TT_Nov24 | 1 | Replaced with 0 |
| Min_TT_Dec24 | 2021 | Replaced with 0 |
| Min_TT_Jan25 | 4000 | Replaced with 0 |
| Min_HorsTT_Nov24 | 1 | Replaced with 0 |
| Min_HorsTT_Dec24 | 2021 | Replaced with 0 |
| Min_HorsTT_Jan25 | 4000 | Replaced with 0 |
| Min_TT_ServiceClient_Nov24 | 1 | Replaced with 0 |
| Min_TT_ServiceClient_Dec24 | 2021 | Replaced with 0 |
| Min_TT_ServiceClient_Jan25 | 4000 | Replaced with 0 |

### Negative Value Treatment Details

| Column | Negative Count | Method |
|--------|---------------|--------|
| ARPU_Nov24 | 9 | Capped extreme negative values |
| ARPU_Dec24 | 15 | Capped extreme negative values |
| ARPU_Jan25 | 5 | Capped extreme negative values |

### Top Columns with Outliers (IQR Method)

| Column | Total Outliers | Lower Bound | Upper Bound | Q1 | Q3 | IQR |
|--------|---------------|-------------|-------------|----|----|----|
| Min_TT_Dec24 | 11551 | -154.43 | 272.96 | 5.84 | 112.69 | 106.85 |
| Min_TT_Jan25 | 11523 | -150.02 | 262.38 | 4.63 | 107.73 | 103.10 |
| Min_TT_Nov24 | 11200 | -159.90 | 286.12 | 7.36 | 118.86 | 111.50 |
| Min_HorsTT_Jan25 | 8314 | -261.90 | 504.94 | 25.66 | 217.37 | 191.71 |
| Min_HorsTT_Dec24 | 8277 | -260.07 | 513.13 | 29.88 | 223.18 | 193.30 |
| Min_HorsTT_Nov24 | 7947 | -261.00 | 527.51 | 34.69 | 231.82 | 197.13 |
| Nbr_Recharges_Jan25 | 5992 | -6.00 | 18.00 | 3.00 | 9.00 | 6.00 |
| Nbr_Recharges_Dec24 | 5570 | -5.00 | 19.00 | 4.00 | 10.00 | 6.00 |
| ARPU_Nov24 | 5409 | -314.35 | 795.02 | 101.66 | 379.00 | 277.34 |
| Nbr_Recharges_Nov24 | 5407 | -5.00 | 19.00 | 4.00 | 10.00 | 6.00 |

## Next Steps

1. Apply feature engineering based on existing EDA insights
2. Build churn prediction models
3. Validate model performance
4. Deploy prediction system
