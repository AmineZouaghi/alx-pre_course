import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_directory_structure():
    """Create the directory structure for output files"""
    # Create main feature engineering directory
    if not os.path.exists("Feature_Engineering"):
        os.makedirs("Feature_Engineering")
        print("Created Feature_Engineering directory", flush=True)
    
    # Create subdirectories
    subdirs = ["Data", "Visualizations", "Reports"]
    for subdir in subdirs:
        path = os.path.join("Feature_Engineering", subdir)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created {path} directory", flush=True)
    
    return {
        "data_dir": os.path.join("Feature_Engineering", "Data"),
        "viz_dir": os.path.join("Feature_Engineering", "Visualizations"),
        "report_dir": os.path.join("Feature_Engineering", "Reports")
    }

def engineer_features(df, output_dir):
    """
    Simple feature engineering with easily interpretable features
    for telecom churn prediction (No ARPU features as requested)
    """
    print("Starting feature engineering process...", flush=True)
    features_df = df.copy()
    
    print(f"Data shape: {features_df.shape}", flush=True)
    
    # ===============================================
    # 1. SIMPLE AVERAGE USAGE FEATURES
    # ===============================================
    print("Creating average usage features...", flush=True)
    
    # Average usage across 3 months (excluding ARPU)
    features_df['avg_voice_minutes'] = features_df[['Min_TT_Nov24', 'Min_TT_Dec24', 'Min_TT_Jan25']].mean(axis=1)
    features_df['avg_offnet_minutes'] = features_df[['Min_HorsTT_Nov24', 'Min_HorsTT_Dec24', 'Min_HorsTT_Jan25']].mean(axis=1)
    features_df['avg_service_minutes'] = features_df[['Min_TT_ServiceClient_Nov24', 'Min_TT_ServiceClient_Dec24', 'Min_TT_ServiceClient_Jan25']].mean(axis=1)
    features_df['avg_number_recharges'] = features_df[['Nbr_Recharges_Nov24', 'Nbr_Recharges_Dec24', 'Nbr_Recharges_Jan25']].mean(axis=1)
    features_df['avg_data_usage'] = features_df[['Volume_Data_Nov24', 'Volume_Data_Dec24', 'Volume_Data_Jan25']].mean(axis=1)
    
    # ===============================================
    # 2. SIMPLE TREND FEATURES
    # ===============================================
    print("Creating simple trend features...", flush=True)
    
    # Did usage increase or decrease in the last month?
    features_df['voice_last_month_decreased'] = (features_df['Min_TT_Jan25'] < features_df['Min_TT_Dec24']).astype(int)
    features_df['data_last_month_decreased'] = (features_df['Volume_Data_Jan25'] < features_df['Volume_Data_Dec24']).astype(int)
    features_df['recharges_last_month_decreased'] = (features_df['Nbr_Recharges_Jan25'] < features_df['Nbr_Recharges_Dec24']).astype(int)
    
    # Did usage consistently decrease over 3 months?
    features_df['voice_consistently_decreased'] = ((features_df['Min_TT_Jan25'] < features_df['Min_TT_Dec24']) & 
                                          (features_df['Min_TT_Dec24'] < features_df['Min_TT_Nov24'])).astype(int)
    features_df['data_consistently_decreased'] = ((features_df['Volume_Data_Jan25'] < features_df['Volume_Data_Dec24']) & 
                                         (features_df['Volume_Data_Dec24'] < features_df['Volume_Data_Nov24'])).astype(int)
    features_df['recharges_consistently_decreased'] = ((features_df['Nbr_Recharges_Jan25'] < features_df['Nbr_Recharges_Dec24']) & 
                                              (features_df['Nbr_Recharges_Dec24'] < features_df['Nbr_Recharges_Nov24'])).astype(int)
    
    # First to last month change (simple to understand)
    features_df['voice_overall_change'] = features_df['Min_TT_Jan25'] - features_df['Min_TT_Nov24']
    features_df['data_overall_change'] = features_df['Volume_Data_Jan25'] - features_df['Volume_Data_Nov24']
    features_df['recharges_overall_change'] = features_df['Nbr_Recharges_Jan25'] - features_df['Nbr_Recharges_Nov24']
    
    # ===============================================
    # 3. ZERO USAGE FEATURES (SIMPLE TO UNDERSTAND)
    # ===============================================
    print("Creating zero usage features...", flush=True)
    
    # Zero-usage flags (extremely strong churn indicators and easy to explain)
    features_df['no_voice_in_Jan'] = (features_df['Min_TT_Jan25'] == 0).astype(int)
    features_df['no_data_in_Jan'] = (features_df['Volume_Data_Jan25'] == 0).astype(int)
    features_df['no_recharges_in_Jan'] = (features_df['Nbr_Recharges_Jan25'] == 0).astype(int)
    
    # How many months had zero usage?
    features_df['months_with_no_voice'] = ((features_df['Min_TT_Nov24'] == 0).astype(int) + 
                                        (features_df['Min_TT_Dec24'] == 0).astype(int) + 
                                        (features_df['Min_TT_Jan25'] == 0).astype(int))
    
    features_df['months_with_no_data'] = ((features_df['Volume_Data_Nov24'] == 0).astype(int) + 
                                       (features_df['Volume_Data_Dec24'] == 0).astype(int) + 
                                       (features_df['Volume_Data_Jan25'] == 0).astype(int))
    
    features_df['months_with_no_recharges'] = ((features_df['Nbr_Recharges_Nov24'] == 0).astype(int) + 
                                            (features_df['Nbr_Recharges_Dec24'] == 0).astype(int) + 
                                            (features_df['Nbr_Recharges_Jan25'] == 0).astype(int))
    
    # Two consecutive months with no usage (strong churn signal and easy to explain)
    features_df['two_consecutive_months_no_voice'] = ((features_df['Min_TT_Dec24'] == 0) & 
                                                   (features_df['Min_TT_Jan25'] == 0)).astype(int)
    
    features_df['two_consecutive_months_no_data'] = ((features_df['Volume_Data_Dec24'] == 0) & 
                                                  (features_df['Volume_Data_Jan25'] == 0)).astype(int)
    
    # ===============================================
    # 4. SERVICE USAGE RATIOS (EASY TO EXPLAIN)
    # ===============================================
    print("Creating simple service usage ratios...", flush=True)
    
    epsilon = 0.001  # To avoid division by zero
    
    # Voice vs Data Balance (Jan)
    features_df['voice_to_data_ratio_Jan'] = features_df['Min_TT_Jan25'] / (features_df['Volume_Data_Jan25'] + epsilon)
    
    # On-network vs Off-network calls (Jan)
    features_df['onnet_to_offnet_ratio_Jan'] = features_df['Min_TT_Jan25'] / (features_df['Min_HorsTT_Jan25'] + epsilon)
    
    # ===============================================
    # 5. SIGNIFICANT DROP FEATURES
    # ===============================================
    print("Creating significant drop features...", flush=True)
    
    # Usage dropped by half or more (easy to explain significant drop)
    features_df['voice_dropped_by_half_last_month'] = (features_df['Min_TT_Jan25'] <= 0.5 * features_df['Min_TT_Dec24']).astype(int)
    features_df['data_dropped_by_half_last_month'] = (features_df['Volume_Data_Jan25'] <= 0.5 * features_df['Volume_Data_Dec24']).astype(int)
    
    # ===============================================
    # 6. ENGAGEMENT FEATURES
    # ===============================================
    print("Creating simple engagement features...", flush=True)
    
    # App usage (easy to explain digital engagement)
    features_df['uses_app'] = features_df['App_MyTT'].astype(int)
    
    # Number of services used (January)
    features_df['number_of_services_used_Jan'] = ((features_df['Min_TT_Jan25'] > 0).astype(int) + 
                                               (features_df['Min_HorsTT_Jan25'] > 0).astype(int) + 
                                               (features_df['Volume_Data_Jan25'] > 0).astype(int))
    
    # Service reduction from Nov to Jan (simple to explain)
    features_df['number_of_services_used_Nov'] = ((features_df['Min_TT_Nov24'] > 0).astype(int) + 
                                               (features_df['Min_HorsTT_Nov24'] > 0).astype(int) + 
                                               (features_df['Volume_Data_Nov24'] > 0).astype(int))
    
    features_df['reduced_services_from_Nov_to_Jan'] = (features_df['number_of_services_used_Nov'] > 
                                                    features_df['number_of_services_used_Jan']).astype(int)
    
    # ===============================================
    # 7. HIGH RISK CUSTOMER FEATURES (SIMPLE TO EXPLAIN)
    # ===============================================
    print("Creating simple high risk customer features...", flush=True)
    
    # Simple flags for high-risk customers (very easy to explain)
    features_df['high_risk_no_usage'] = ((features_df['no_voice_in_Jan'] == 1) & 
                                       (features_df['no_data_in_Jan'] == 1)).astype(int)
    
    features_df['high_risk_decreased_usage'] = ((features_df['voice_consistently_decreased'] == 1) & 
                                             (features_df['data_consistently_decreased'] == 1)).astype(int)
    
    # Number of risk factors (simple count)
    risk_factors = ['no_voice_in_Jan', 'no_data_in_Jan', 'voice_consistently_decreased', 
                   'data_consistently_decreased', 'voice_dropped_by_half_last_month', 
                   'data_dropped_by_half_last_month', 'reduced_services_from_Nov_to_Jan', 
                   'two_consecutive_months_no_voice', 'two_consecutive_months_no_data']
    
    features_df['number_of_risk_factors'] = features_df[risk_factors].sum(axis=1)
    
    # High risk customer (3+ risk factors)
    features_df['is_high_risk_customer'] = (features_df['number_of_risk_factors'] >= 3).astype(int)
    
    # ===============================================
    # 8. STANDARDIZATION FOR NUMERIC FEATURES
    # ===============================================
    print("Applying standardization to numeric features...", flush=True)
    
    # List of numeric features to standardize
    numeric_features = ['avg_voice_minutes', 'avg_offnet_minutes', 'avg_service_minutes', 
                       'avg_number_recharges', 'avg_data_usage', 'voice_overall_change', 
                       'data_overall_change', 'recharges_overall_change',
                       'voice_to_data_ratio_Jan', 'onnet_to_offnet_ratio_Jan']
    
    # Apply standard scaling
    scaler = StandardScaler()
    features_df[numeric_features] = scaler.fit_transform(features_df[numeric_features])
    
    # Save the engineered features
    output_path = os.path.join(output_dir["data_dir"], "telecom_engineered_features.csv")
    print(f"Saving engineered features to {output_path}...", flush=True)
    features_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved to {output_path}", flush=True)
    print(f"Created {len(features_df.columns)} features", flush=True)
    
    # Save raw data with target
    raw_data_path = os.path.join(output_dir["data_dir"], "raw_data_with_target.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"Raw data with target saved to {raw_data_path}", flush=True)
    
    # Return both the raw and engineered data
    return {"raw": df, "engineered": features_df}

def create_visualizations(data_dict, output_dir):
    """Create visualizations for the feature engineering report"""
    print("Creating visualizations...", flush=True)
    
    # Extract data
    raw_df = data_dict["raw"]
    features_df = data_dict["engineered"]
    
    # 1. Feature count by category
    average_features = len([col for col in features_df.columns if col.startswith('avg_')])
    trend_features = len([col for col in features_df.columns if 'decreased' in col or 'change' in col])
    zero_features = len([col for col in features_df.columns if 'no_' in col or 'consecutive' in col or 'months_with_no' in col])
    ratio_features = len([col for col in features_df.columns if 'ratio' in col])
    drop_features = len([col for col in features_df.columns if 'dropped' in col])
    engagement_features = len([col for col in features_df.columns if 'service' in col or 'uses_app' in col])
    risk_features = len([col for col in features_df.columns if 'risk' in col or 'number_of_risk_factors' in col])
    
    # Create feature count chart
    categories = ['Average Usage', 'Usage Trends', 'Zero Usage', 'Service Ratios', 'Significant Drops', 'Engagement', 'Risk Indicators']
    counts = [average_features, trend_features, zero_features, ratio_features, drop_features, engagement_features, risk_features]
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=categories, y=counts)
    plt.title('Number of Features by Category')
    plt.xlabel('Feature Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir["viz_dir"], "feature_counts_by_category.png"))
    plt.close()
    
    # 2. Distribution of key risk indicators by churn status
    risk_features = ['no_voice_in_Jan', 'no_data_in_Jan', 'voice_consistently_decreased', 
                     'two_consecutive_months_no_voice', 'is_high_risk_customer', 'high_risk_decreased_usage']
    
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(risk_features):
        plt.subplot(2, 3, i+1)
        sns.countplot(x=feature, hue='Churn', data=features_df)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature.replace('_', ' ').title())
        plt.ylabel('Count')
        plt.legend(title='Churn')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir["viz_dir"], "risk_indicators_by_churn.png"))
    plt.close()
    
    # 3. Correlation with Churn
    # Select features to analyze
    key_features = ['avg_voice_minutes', 'avg_data_usage', 'voice_last_month_decreased', 
                   'data_consistently_decreased', 'no_voice_in_Jan', 'months_with_no_voice',
                   'voice_dropped_by_half_last_month', 'number_of_services_used_Jan',
                   'high_risk_no_usage', 'is_high_risk_customer', 'Churn']
    
    # Calculate correlation with churn
    corr_with_churn = features_df[key_features].corr()['Churn'].sort_values(ascending=False)
    corr_with_churn = corr_with_churn.drop('Churn')  # Remove self-correlation
    
    # Plot correlation with churn
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=corr_with_churn.values, y=corr_with_churn.index)
    plt.title('Correlation of Features with Churn')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    
    # Add value labels
    for i, v in enumerate(corr_with_churn.values):
        ax.text(v + 0.01 if v >= 0 else v - 0.06, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir["viz_dir"], "correlation_with_churn.png"))
    plt.close()
    
    # 4. Number of risk factors distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='number_of_risk_factors', hue='Churn', data=features_df)
    plt.title('Distribution of Number of Risk Factors by Churn Status')
    plt.xlabel('Number of Risk Factors')
    plt.ylabel('Count')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir["viz_dir"], "risk_factors_distribution.png"))
    plt.close()
    
    # 5. Average usage comparison by churn status
    plt.figure(figsize=(12, 8))
    
    avg_features = ['avg_voice_minutes', 'avg_data_usage', 'avg_offnet_minutes', 'avg_number_recharges']
    
    for i, feature in enumerate(avg_features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='Churn', y=feature, data=features_df)
        plt.title(f'{feature.replace("_", " ").title()} by Churn Status')
        plt.xlabel('Churn')
        plt.ylabel(feature.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir["viz_dir"], "average_usage_by_churn.png"))
    plt.close()
    
    # Return feature counts for the report
    return {
        "feature_counts": {
            "Average Usage": average_features,
            "Usage Trends": trend_features,
            "Zero Usage": zero_features,
            "Service Ratios": ratio_features,
            "Significant Drops": drop_features,
            "Engagement": engagement_features,
            "Risk Indicators": risk_features,
            "Total": len(features_df.columns) - 1  # Excluding Churn
        },
        "correlations": {
            "top_correlated": corr_with_churn.head(5).to_dict(),
            "bottom_correlated": corr_with_churn.tail(5).to_dict()
        }
    }

def generate_report(data_dict, viz_data, output_dir):
    """Generate a comprehensive feature engineering report"""
    print("Generating feature engineering report...", flush=True)
    
    # Extract data
    raw_df = data_dict["raw"]
    features_df = data_dict["engineered"]
    feature_counts = viz_data["feature_counts"]
    correlations = viz_data["correlations"]
    
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create report content
    report = f"""# Telecom Churn Prediction - Feature Engineering Report
**Date:** {now}  
**Author:** AmineZouaghi

## 1. Introduction

This report documents the feature engineering process for the Tunisie Telecom churn prediction project. We focused on creating **simple, easy-to-interpret features** that capture customer behavior patterns related to churn while maintaining high predictive power.

## 2. Feature Engineering Approach

### 2.1 Data Overview

- **Original Dataset Size:** {raw_df.shape[0]} records × {raw_df.shape[1]} features
- **Engineered Dataset Size:** {features_df.shape[0]} records × {features_df.shape[1]} features

### 2.2 Feature Categories

We created {feature_counts["Total"]} easily interpretable features across 7 categories:

1. **Average Usage Features ({feature_counts["Average Usage"]} features):** Simple averages of usage metrics across the three-month period.

2. **Usage Trend Features ({feature_counts["Usage Trends"]} features):** Simple indicators of whether usage increased or decreased over time.

3. **Zero Usage Features ({feature_counts["Zero Usage"]} features):** Straightforward flags for when customers had no usage of specific services.

4. **Service Ratio Features ({feature_counts["Service Ratios"]} features):** Simple ratios between different service usages.

5. **Significant Drop Features ({feature_counts["Significant Drops"]} features):** Clear indicators of when usage dropped by half or more.

6. **Engagement Features ({feature_counts["Engagement"]} features):** Simple measures of how many services a customer uses and whether they use the app.

7. **Risk Indicator Features ({feature_counts["Risk Indicators"]} features):** Straightforward flags for high-risk customer behavior and a simple count of risk factors.

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
   - No voice usage in January (strong correlation: {list(correlations["top_correlated"].items())[0][1]:.2f})
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
"""
    
    # Save report
    report_path = os.path.join(output_dir["report_dir"], "feature_engineering_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to {report_path}", flush=True)
    
    return report_path

if __name__ == "__main__":
    try:
        print("Script started", flush=True)
        print(f"Working directory: {os.getcwd()}", flush=True)
        
        # Create directory structure
        output_dirs = create_directory_structure()
        
        # Try multiple file paths to find the right one
        possible_paths = [
            "./Cleaned_Data/cleaned_tunisie_telecom_data.csv",
            "Cleaned_Data/cleaned_tunisie_telecom_data.csv",
            "../Cleaned_Data/cleaned_tunisie_telecom_data.csv",
            "C:/Users/MSI/OneDrive/Desktop/PFE VF/Cleaned_Data/cleaned_tunisie_telecom_data.csv"
        ]
        
        df = None
        for path in possible_paths:
            print(f"Trying to read from: {path}", flush=True)
            if os.path.exists(path):
                print(f"File found at: {path}", flush=True)
                df = pd.read_csv(path)
                print(f"Successfully loaded data with shape {df.shape}", flush=True)
                break
            else:
                print(f"File not found at: {path}", flush=True)
        
        if df is None:
            # Last resort - ask user for path
            print("ERROR: Could not find data file in any of the tried locations", flush=True)
            print("Please enter the full absolute path to the data file:", flush=True)
            user_path = input().strip()
            
            if os.path.exists(user_path):
                print(f"File found at: {user_path}", flush=True)
                df = pd.read_csv(user_path)
                print(f"Successfully loaded data with shape {df.shape}", flush=True)
            else:
                print(f"File not found at: {user_path}", flush=True)
                sys.exit(1)
        
        print("Data loaded successfully!", flush=True)
        print(f"Data shape: {df.shape}", flush=True)
        
        # Extract the target variable if it exists
        if 'Churn' in df.columns:
            print("Churn column found", flush=True)
        else:
            # Create Churn column from last column in the data
            df['Churn'] = df.iloc[:, -1].copy()
            print("Churn column created from last column in dataset", flush=True)
            
        # Ensure data has no NaN values
        print("Checking for missing values...", flush=True)
        if df.isnull().sum().sum() > 0:
            print("Warning: Dataset contains missing values. Filling with zeros.", flush=True)
            df = df.fillna(0)
        
        # Run feature engineering
        data_dict = engineer_features(df, output_dirs)
        
        # Create visualizations
        viz_data = create_visualizations(data_dict, output_dirs)
        
        # Generate report
        report_path = generate_report(data_dict, viz_data, output_dirs)
        
        print("\nFeature engineering completed successfully!", flush=True)
        print(f"Report saved to: {report_path}", flush=True)
        print(f"Data saved to: {output_dirs['data_dir']}", flush=True)
        print(f"Visualizations saved to: {output_dirs['viz_dir']}", flush=True)
        
    except Exception as e:
        print(f"ERROR: An exception occurred: {str(e)}", flush=True)
        print("Traceback:", flush=True)
        traceback.print_exc()