# Telecom Churn Prediction - Feature Engineering Report
**Date:** 2025-06-17 18:42:35  
**Author:** AmineZouaghi

## 1. Introduction

This report documents the feature engineering process for the Tunisie Telecom churn prediction project. We focused on creating **simple, easy-to-interpret features** that capture customer behavior patterns related to churn while maintaining high predictive power.

## 2. Feature Engineering Approach

### 2.1 Data Overview

- **Original Dataset Size:** 92943 records × 22 features
- **Engineered Dataset Size:** 92943 records × 56 features

### 2.2 Feature Categories

We created 55 easily interpretable features across 7 categories:

1. **Average Usage Features (5 features):** Simple averages of usage metrics across the three-month period.

2. **Usage Trend Features (10 features):** Simple indicators of whether usage increased or decreased over time.

3. **Zero Usage Features (9 features):** Straightforward flags for when customers had no usage of specific services.

4. **Service Ratio Features (2 features):** Simple ratios between different service usages.

5. **Significant Drop Features (2 features):** Clear indicators of when usage dropped by half or more.

6. **Engagement Features (5 features):** Simple measures of how many services a customer uses and whether they use the app.

7. **Risk Indicator Features (['no_voice_in_Jan', 'no_data_in_Jan', 'voice_consistently_decreased', 'two_consecutive_months_no_voice', 'is_high_risk_customer', 'high_risk_decreased_usage'] features):** Straightforward flags for high-risk customer behavior and a simple count of risk factors.

### 2.3 Methodology

Our approach focused on creating features that are:

1. **Easy to Interpret:** All features have clear, intuitive meanings that can be easily explained.

2. **Business-Relevant:** Each feature directly relates to customer behavior that business stakeholders understand.

3. **Simple Transformations:** We used simple operations like averages, counts, and binary flags rather than complex formulas.

4. **No ARPU Dependency:** As requested, we excluded ARPU as an input feature, focusing instead on direct usage metrics.

## 3. Feature Engineering Highlights

### 3.1 Feature Count by Category

![Feature Counts by Category](Feature_Engineering/Visualizations/feature_counts_by_category.png)

### 3.2 Key Risk Indicators by Churn Status

![Risk Indicators by Churn Status](Feature_Engineering/Visualizations/risk_indicators_by_churn.png)

### 3.3 Correlation of Features with Churn

![Correlation with Churn](Feature_Engineering/Visualizations/correlation_with_churn.png)

### 3.4 Number of Risk Factors Distribution

![Risk Factors Distribution](Feature_Engineering/Visualizations/risk_factors_distribution.png)

### 3.5 Average Usage by Churn Status

![Average Usage by Churn](Feature_Engineering/Visualizations/average_usage_by_churn.png)

## 4. Key Findings

1. **Top Churn Predictors:**
   - No voice usage in January (strong correlation: 0.36)
   - Two consecutive months with no voice usage
   - High-risk customer flag (customers with 3+ risk factors)
   - Consistently decreasing voice and data usage

2. **Usage Patterns:**
   - Churners show significantly lower average usage across all services
   - Churners are more likely to have zero usage in the final month
   - The number of risk factors is strongly correlated with churn probability

3. **Simple But Effective:** 
   - Even with simple features, we can effectively identify high-risk customers
   - The "is_high_risk_customer" feature (3+ risk factors) is particularly powerful

## 5. How to Interpret Key Features

1. **no_voice_in_Jan:** Flag (0 or 1) indicating whether the customer had zero voice minutes in January.

2. **months_with_no_voice:** Simple count (0-3) of how many months the customer had zero voice usage.

3. **voice_consistently_decreased:** Flag indicating whether voice usage decreased each month (Nov>Dec>Jan).

4. **voice_dropped_by_half_last_month:** Flag indicating if voice usage in January was less than half of December's usage.

5. **reduced_services_from_Nov_to_Jan:** Flag showing whether the customer is using fewer services in January than in November.

6. **number_of_risk_factors:** Simple count of how many risk factors apply to each customer.

7. **is_high_risk_customer:** Flag identifying customers with 3 or more risk factors.

## 6. Next Steps

1. **Model Training:** Use these interpretable features to train churn prediction models.

2. **Business Integration:** Present these features to business teams for better understanding of churn drivers.

3. **Feature Importance Analysis:** Identify which simple features contribute most to model performance.

4. **ARPU Analysis:** As requested, conduct separate analysis using ARPU as an external variable.

## 7. Files and Directory Structure

- **Data Files:**
  - Raw data with target: `Feature_Engineering/Data/raw_data_with_target.csv`
  - Engineered features: `Feature_Engineering/Data/telecom_engineered_features.csv`

- **Visualizations:**
  - Feature counts: `Feature_Engineering/Visualizations/feature_counts_by_category.png`
  - Risk indicators: `Feature_Engineering/Visualizations/risk_indicators_by_churn.png`
  - Correlation with churn: `Feature_Engineering/Visualizations/correlation_with_churn.png`
  - Risk factors distribution: `Feature_Engineering/Visualizations/risk_factors_distribution.png`
  - Average usage by churn: `Feature_Engineering/Visualizations/average_usage_by_churn.png`

- **Reports:**
  - This report: `Feature_Engineering/Reports/feature_engineering_report.md`
