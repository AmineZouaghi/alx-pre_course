# Data Dictionary

## Original Features

### App_MyTT
Binary indicator of whether customer has MyTT app installed (1=Yes, 0=No)

### ARPU_Nov24
Average Revenue Per User for November 2024

### ARPU_Dec24
Average Revenue Per User for December 2024

### ARPU_Jan25
Average Revenue Per User for January 2025

### Min_TT_Nov24
On-net voice minutes for November 2024

### Min_TT_Dec24
On-net voice minutes for December 2024

### Min_TT_Jan25
On-net voice minutes for January 2025

### Min_HorsTT_Nov24
Off-net voice minutes for November 2024

### Min_HorsTT_Dec24
Off-net voice minutes for December 2024

### Min_HorsTT_Jan25
Off-net voice minutes for January 2025

### Min_TT_ServiceClient_Nov24
Minutes spent calling customer service in November 2024

### Min_TT_ServiceClient_Dec24
Minutes spent calling customer service in December 2024

### Min_TT_ServiceClient_Jan25
Minutes spent calling customer service in January 2025

### Nbr_Recharges_Nov24
Number of recharges in November 2024

### Nbr_Recharges_Dec24
Number of recharges in December 2024

### Nbr_Recharges_Jan25
Number of recharges in January 2025

### Volume_Data_Nov24
Data usage volume for November 2024

### Volume_Data_Dec24
Data usage volume for December 2024

### Volume_Data_Jan25
Data usage volume for January 2025

### Ancienneté_Abonné
Customer tenure in days

### Résiliation
Churn indicator (1=Churned, 0=Retained)

### Tenure_Bucket
Categorized tenure in groups: <6 months, 6-12 months, 12-24 months, >24 months

### Has_Service_Calls
Binary indicator of whether customer called customer service in observation period

## Derived Features

### ARPU_Trend_Nov_Jan
Percentage change in ARPU from November to January

### ARPU_Last_Change
Percentage change in ARPU from December to January (last month)

### Min_TT_Trend
Percentage change in on-net minutes from November to January

### Min_HorsTT_Trend
Percentage change in off-net minutes from November to January

### Data_Trend
Percentage change in data volume from November to January

### Off_Net_Ratio
Ratio of off-net minutes to total minutes (higher values indicate more off-net usage)

