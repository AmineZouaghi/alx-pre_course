import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set configuration with user-specified values
CURRENT_DATE = "2025-06-17 04:51:30"  # As specified by user
CURRENT_USER = "AmineZouaghi"  # As specified by user
INPUT_FILE = "Data Tunisie Telecom.csv"
OUTPUT_DIR = "Cleaned_Data"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Create output directories
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Set up simple logging
LOG_FILE = f"{OUTPUT_DIR}/cleaning_log.txt"
def log_message(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{log_entry}\n")

# Step 1: Load the data with proper encoding
log_message("Loading data...")
try:
    # Try different encodings since there were character issues
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(INPUT_FILE, encoding=encoding)
            log_message(f"Successfully loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log_message(f"Error with {encoding}: {str(e)}")
    
    # Save a copy of the original data
    df_original = df.copy()
    log_message(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Initial data diagnosis
    log_message("Performing initial data diagnosis...")
    
    # Count missing values by column
    missing_by_col = df.isna().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    if not missing_cols.empty:
        log_message(f"Found missing values in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            log_message(f"  - {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    neg_values_by_col = {}
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            neg_values_by_col[col] = neg_count
    
    if neg_values_by_col:
        log_message(f"Found negative values in {len(neg_values_by_col)} columns:")
        for col, count in neg_values_by_col.items():
            log_message(f"  - {col}: {count} negative values ({count/len(df)*100:.2f}%)")
    
    # Step 2: Check and remove duplicates
    log_message("Checking for duplicates...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        log_message(f"Found {duplicates} duplicate rows. Removing them...")
        df = df.drop_duplicates().reset_index(drop=True)
        log_message(f"After removing duplicates: {df.shape[0]} rows")
    else:
        log_message("No duplicates found")
    
    # Step 3: Handle missing values
    log_message("Handling missing values...")
    missing_before = df.isna().sum().sum()
    log_message(f"Missing values before: {missing_before}")
    
    missing_fix_details = {}
    
    # For usage metrics (minutes, data, recharges), replace NaN with 0
    usage_cols = [col for col in df.columns if any(x in col for x in ['Min_', 'Volume_Data_', 'Nbr_Recharges_'])]
    for col in usage_cols:
        if df[col].isna().sum() > 0:
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            missing_fix_details[col] = {"method": "fill_zero", "count": missing_count}
    
    # For ARPU columns, use median
    arpu_cols = [col for col in df.columns if 'ARPU_' in col]
    for col in arpu_cols:
        if df[col].isna().sum() > 0:
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            missing_fix_details[col] = {"method": "fill_median", "count": missing_count}
    
    # For binary columns, fill with 0
    binary_cols = ['App_MyTT']
    
    # Find churn column
    churn_col = None
    for col in df.columns:
        if col in ['Résiliation', 'Resiliation'] or 'sil' in col.lower():
            churn_col = col
            binary_cols.append(col)
            break
    
    for col in binary_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(0).astype(int)
            missing_fix_details[col] = {"method": "fill_zero_binary", "count": missing_count}
    
    # For any remaining numeric columns, use median
    for col in numeric_cols:
        if col not in usage_cols + arpu_cols + binary_cols and df[col].isna().sum() > 0:
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            missing_fix_details[col] = {"method": "fill_median", "count": missing_count}
    
    # For any remaining columns, use mode
    remaining_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    for col in remaining_cols:
        missing_count = df[col].isna().sum()
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        missing_fix_details[col] = {"method": "fill_mode", "count": missing_count}
    
    missing_after = df.isna().sum().sum()
    missing_fixed = missing_before - missing_after
    log_message(f"Missing values after: {missing_after}")
    log_message(f"Fixed {missing_fixed} missing values")
    
    # Step 4: Fix negative values in all relevant columns
    log_message("Fixing negative values...")
    
    negative_fix_details = {}
    
    # Fix negative values in usage metrics
    for col in usage_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            df[col] = df[col].clip(lower=0)
            negative_fix_details[col] = {"count": neg_count, "method": "clip_to_zero"}
    
    # Handle negative values in other metrics (except those where negative may be valid)
    exclude_from_neg_check = binary_cols.copy()
    if churn_col:
        exclude_from_neg_check.append(churn_col)
    
    # For ARPU columns, only fix extreme negative values (might be valid to be slightly negative)
    for col in arpu_cols:
        extreme_neg_count = (df[col] < -100).sum()  # Only fix extremely negative values
        if extreme_neg_count > 0:
            df.loc[df[col] < -100, col] = -100  # Cap at reasonable negative value
            negative_fix_details[col] = {"count": extreme_neg_count, "method": "cap_extreme_negative"}
    
    # Check other numeric columns
    other_numeric = [col for col in numeric_cols if col not in usage_cols + arpu_cols + exclude_from_neg_check]
    for col in other_numeric:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            # If it's a count-like column, clip to 0
            if any(word in col.lower() for word in ['count', 'number', 'nbr', 'qty']):
                df[col] = df[col].clip(lower=0)
                negative_fix_details[col] = {"count": neg_count, "method": "clip_to_zero"}
            # Otherwise, just track but don't change
            else:
                negative_fix_details[col] = {"count": neg_count, "method": "no_change"}
    
    total_neg_fixed = sum(details["count"] for col, details in negative_fix_details.items() 
                        if details["method"] != "no_change")
    log_message(f"Fixed {total_neg_fixed} negative values across {len(negative_fix_details)} columns")
    
    # Step 5: Handle outliers using IQR method
    log_message("Handling outliers using IQR method...")
    
    outlier_fix_details = {}
    total_outliers_fixed = 0
    
    # Store pre-cleaning distributions for visualization
    distributions_before = {}
    
    # Generate IQR cutoffs and distribution data
    for col in numeric_cols:
        # Skip binary columns and columns with few unique values
        if df[col].nunique() <= 5 or col in binary_cols:
            continue
            
        # Store pre-cleaning data and statistics
        distributions_before[col] = df[col].copy()
        
        # Calculate IQR bounds - only if there are sufficient non-missing values
        if df[col].count() > 10:  # Ensure enough data for meaningful quartiles
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Verify IQR is not zero or very small
            if IQR > 1e-10:  # Avoid division by zero or very small IQR
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count and fix outliers
                lower_outliers = (df[col] < lower_bound).sum()
                upper_outliers = (df[col] > upper_bound).sum()
                total_outliers = lower_outliers + upper_outliers
                
                if total_outliers > 0:
                    # Cap outliers at bounds
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    
                    outlier_fix_details[col] = {
                        "count": total_outliers,
                        "lower_count": lower_outliers,
                        "upper_count": upper_outliers,
                        "method": "IQR",
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "q1": Q1,
                        "q3": Q3,
                        "iqr": IQR
                    }
                    
                    total_outliers_fixed += total_outliers
    
    log_message(f"Capped {total_outliers_fixed} outliers using IQR method across {len(outlier_fix_details)} columns")
    
    # Step 6: Ensure binary columns are really binary
    log_message("Standardizing binary columns...")
    
    binary_standardized = {}
    for col in binary_cols:
        if col in df.columns:
            before_unique = df[col].nunique()
            df[col] = df[col].map(lambda x: 1 if x not in [0, np.nan] else 0).astype(int)
            after_unique = df[col].nunique()
            if before_unique != after_unique:
                binary_standardized[col] = {"before": before_unique, "after": after_unique}
    
    if binary_standardized:
        log_message(f"Standardized {len(binary_standardized)} binary columns")
    else:
        log_message("Binary columns already standardized")
    
    # Step 7: Save the cleaned data
    output_file = f"{OUTPUT_DIR}/cleaned_tunisie_telecom_data.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Step 8: Generate distribution plots for HTML report
    log_message("Generating distribution plots for report...")
    
    # Get top 10 columns with most outliers
    top_outlier_cols = sorted(outlier_fix_details.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    
    # Prepare plot data for report
    plot_data = {}
    
    for i, (col, details) in enumerate(top_outlier_cols):
        # Create figure with before/after distributions
        plt.figure(figsize=(14, 6))
        
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before cleaning - original distribution
        sns.histplot(distributions_before[col], kde=True, ax=ax1)
        ax1.set_title(f'Before Cleaning: {col}')
        ax1.axvline(details['lower_bound'], color='r', linestyle='--', label=f'Lower bound: {details["lower_bound"]:.2f}')
        ax1.axvline(details['upper_bound'], color='r', linestyle='--', label=f'Upper bound: {details["upper_bound"]:.2f}')
        ax1.axvline(details['q1'], color='g', linestyle='-', label=f'Q1: {details["q1"]:.2f}')
        ax1.axvline(details['q3'], color='g', linestyle='-', label=f'Q3: {details["q3"]:.2f}')
        ax1.legend()
        
        # After cleaning - capped distribution
        sns.histplot(df[col], kde=True, ax=ax2)
        ax2.set_title(f'After Cleaning: {col}')
        ax2.axvline(details['lower_bound'], color='r', linestyle='--', label=f'Lower bound: {details["lower_bound"]:.2f}')
        ax2.axvline(details['upper_bound'], color='r', linestyle='--', label=f'Upper bound: {details["upper_bound"]:.2f}')
        ax2.axvline(details['q1'], color='g', linestyle='-', label=f'Q1: {details["q1"]:.2f}')
        ax2.axvline(details['q3'], color='g', linestyle='-', label=f'Q3: {details["q3"]:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot to file
        plot_filename = f"{PLOTS_DIR}/distribution_{col.replace(' ', '_')}.png"
        plt.savefig(plot_filename)
        plt.close(fig)
        
        # Store plot data for report
        plot_data[col] = {
            'filename': plot_filename,
            'details': details,
            'before_stats': {
                'min': distributions_before[col].min(),
                'max': distributions_before[col].max(),
                'mean': distributions_before[col].mean(),
                'median': distributions_before[col].median(),
                'std': distributions_before[col].std()
            },
            'after_stats': {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            }
        }
    
    # Step 9: Create Markdown summary
    report_file = f"{OUTPUT_DIR}/cleaning_summary.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Data Cleaning Summary\n\n")
        f.write(f"**Date:** {CURRENT_DATE}\n")
        f.write(f"**User:** {CURRENT_USER}\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"* Original size: {df_original.shape[0]} rows × {df_original.shape[1]} columns\n")
        f.write(f"* Cleaned size: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        
        f.write("## Actions Performed\n\n")
        f.write(f"1. Removed {duplicates} duplicate rows\n")
        f.write(f"2. Fixed {missing_fixed} missing values\n")
        f.write(f"3. Corrected {total_neg_fixed} negative values\n")
        f.write(f"4. Capped {total_outliers_fixed} outliers using IQR method\n")
        f.write(f"5. Standardized binary columns to contain only 0 and 1\n\n")
        
        # Details on missing value fixes
        if missing_fix_details:
            f.write("### Missing Value Treatment Details\n\n")
            f.write("| Column | Missing Count | Method |\n")
            f.write("|--------|--------------|--------|\n")
            for col, details in missing_fix_details.items():
                method_desc = {
                    "fill_zero": "Replaced with 0",
                    "fill_median": "Replaced with median",
                    "fill_mode": "Replaced with mode",
                    "fill_zero_binary": "Replaced with 0 (binary column)"
                }
                f.write(f"| {col} | {details['count']} | {method_desc.get(details['method'], details['method'])} |\n")
            f.write("\n")
        
        # Details on negative value fixes
        if negative_fix_details:
            f.write("### Negative Value Treatment Details\n\n")
            f.write("| Column | Negative Count | Method |\n")
            f.write("|--------|---------------|--------|\n")
            for col, details in negative_fix_details.items():
                method_desc = {
                    "clip_to_zero": "Replaced with 0",
                    "cap_extreme_negative": "Capped extreme negative values",
                    "no_change": "Identified but preserved (may be valid)"
                }
                f.write(f"| {col} | {details['count']} | {method_desc.get(details['method'], details['method'])} |\n")
            f.write("\n")
        
        # Top outlier columns with IQR details
        if outlier_fix_details:
            f.write("### Top Columns with Outliers (IQR Method)\n\n")
            f.write("| Column | Total Outliers | Lower Bound | Upper Bound | Q1 | Q3 | IQR |\n")
            f.write("|--------|---------------|-------------|-------------|----|----|----|\n")
            for col, details in top_outlier_cols:
                f.write(f"| {col} | {details['count']} | {details['lower_bound']:.2f} | {details['upper_bound']:.2f} | {details['q1']:.2f} | {details['q3']:.2f} | {details['iqr']:.2f} |\n")
            f.write("\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Apply feature engineering based on existing EDA insights\n")
        f.write("2. Build churn prediction models\n")
        f.write("3. Validate model performance\n")
        f.write("4. Deploy prediction system\n")
    
    # Step 10: Create detailed HTML report
    html_report_file = f"{OUTPUT_DIR}/rapport.html"
    with open(html_report_file, 'w', encoding='utf-8') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detailed Data Cleaning Report - Tunisie Telecom</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .plot-container {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .plot-image {{
                    width: 100%;
                    max-width: 1000px;
                    margin: 0 auto;
                    display: block;
                }}
                .stats-table {{
                    width: 100%;
                    margin-top: 15px;
                }}
                .improvement {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .important {{
                    font-weight: bold;
                    color: #e74c3c;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    text-align: center;
                    font-size: 0.9em;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Detailed Data Cleaning Report - Tunisie Telecom</h1>
                <p><strong>Date:</strong> {CURRENT_DATE}</p>
                <p><strong>User:</strong> {CURRENT_USER}</p>
                
                <div class="summary-box">
                    <h2>Executive Summary</h2>
                    <p>This report details the data cleaning process applied to the Tunisie Telecom dataset using the IQR (Interquartile Range) method for outlier detection.</p>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                        <tr>
                            <td>Total rows</td>
                            <td>{df.shape[0]}</td>
                            <td>100%</td>
                        </tr>
                        <tr>
                            <td>Missing values fixed</td>
                            <td>{missing_fixed}</td>
                            <td>{missing_fixed/(df.shape[0]*df.shape[1])*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Negative values corrected</td>
                            <td>{total_neg_fixed}</td>
                            <td>{total_neg_fixed/(df.shape[0]*len(numeric_cols))*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Outliers capped (IQR method)</td>
                            <td>{total_outliers_fixed}</td>
                            <td>{total_outliers_fixed/(df.shape[0]*len([c for c in numeric_cols if c not in binary_cols]))*100:.2f}%</td>
                        </tr>
                    </table>
                </div>
                
                <h2>Data Distribution Analysis</h2>
                <p>The following sections show how the data is distributed before and after cleaning for the columns with the most outliers.</p>
                
                <h3>How IQR Method Works</h3>
                <p>The Interquartile Range (IQR) method identifies outliers based on the statistical distribution of the data:</p>
                <ul>
                    <li><strong>Q1 (First Quartile):</strong> 25th percentile of the data</li>
                    <li><strong>Q3 (Third Quartile):</strong> 75th percentile of the data</li>
                    <li><strong>IQR (Interquartile Range):</strong> Q3 - Q1</li>
                    <li><strong>Lower Bound:</strong> Q1 - 1.5 × IQR</li>
                    <li><strong>Upper Bound:</strong> Q3 + 1.5 × IQR</li>
                </ul>
                <p>Any values below the lower bound or above the upper bound are considered outliers and are capped at these thresholds.</p>
        """)
        
        # Add distribution plots and statistics for each column
        for col, data in plot_data.items():
            details = data['details']
            before_stats = data['before_stats']
            after_stats = data['after_stats']
            
            # Calculate improvements
            range_before = before_stats['max'] - before_stats['min']
            range_after = after_stats['max'] - after_stats['min']
            std_reduction = ((before_stats['std'] - after_stats['std']) / before_stats['std']) * 100 if before_stats['std'] > 0 else 0
            
            f.write(f"""
                <div class="plot-container">
                    <h3>Column: {col}</h3>
                    <p><strong>Total Outliers:</strong> {details['count']} ({details['count']/len(df)*100:.2f}% of data)</p>
                    <p><strong>Lower Outliers:</strong> {details['lower_count']} | <strong>Upper Outliers:</strong> {details['upper_count']}</p>
                    
                    <img src="../{data['filename']}" class="plot-image" alt="Distribution plot for {col}">
                    
                    <h4>Statistical Comparison</h4>
                    <table class="stats-table">
                        <tr>
                            <th>Statistic</th>
                            <th>Before Cleaning</th>
                            <th>After Cleaning</th>
                            <th>Change</th>
                        </tr>
                        <tr>
                            <td>Minimum</td>
                            <td>{before_stats['min']:.2f}</td>
                            <td>{after_stats['min']:.2f}</td>
                            <td>{after_stats['min'] - before_stats['min']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Maximum</td>
                            <td>{before_stats['max']:.2f}</td>
                            <td>{after_stats['max']:.2f}</td>
                            <td>{after_stats['max'] - before_stats['max']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Mean</td>
                            <td>{before_stats['mean']:.2f}</td>
                            <td>{after_stats['mean']:.2f}</td>
                            <td>{after_stats['mean'] - before_stats['mean']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Median</td>
                            <td>{before_stats['median']:.2f}</td>
                            <td>{after_stats['median']:.2f}</td>
                            <td>{after_stats['median'] - before_stats['median']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Standard Deviation</td>
                            <td>{before_stats['std']:.2f}</td>
                            <td>{after_stats['std']:.2f}</td>
                            <td class="improvement">↓ {std_reduction:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Range</td>
                            <td>{range_before:.2f}</td>
                            <td>{range_after:.2f}</td>
                            <td class="improvement">↓ {((range_before - range_after) / range_before) * 100 if range_before > 0 else 0:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Q1 (25th percentile)</td>
                            <td>{details['q1']:.2f}</td>
                            <td>{details['q1']:.2f}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>Q3 (75th percentile)</td>
                            <td>{details['q3']:.2f}</td>
                            <td>{details['q3']:.2f}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>IQR</td>
                            <td>{details['iqr']:.2f}</td>
                            <td>{details['iqr']:.2f}</td>
                            <td>-</td>
                        </tr>
                    </table>
                    
                    <h4>Outlier Boundaries</h4>
                    <ul>
                        <li><strong>Lower Bound:</strong> {details['lower_bound']:.2f} (values below this were capped)</li>
                        <li><strong>Upper Bound:</strong> {details['upper_bound']:.2f} (values above this were capped)</li>
                    </ul>
                </div>
            """)
        
        # Add summary tables
        f.write(f"""
                <h2>Cleaning Actions Summary</h2>
                
                <h3>Missing Value Treatment</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Missing Count</th>
                        <th>Method</th>
                    </tr>
        """)
        
        # Add missing value details
        for col, details in missing_fix_details.items():
            method_desc = {
                "fill_zero": "Replaced with 0",
                "fill_median": "Replaced with median",
                "fill_mode": "Replaced with mode",
                "fill_zero_binary": "Replaced with 0 (binary column)"
            }
            f.write(f"""
                    <tr>
                        <td>{col}</td>
                        <td>{details['count']}</td>
                        <td>{method_desc.get(details['method'], details['method'])}</td>
                    </tr>
            """)
        
        # Add negative value details
        f.write(f"""
                </table>
                
                <h3>Negative Value Treatment</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Negative Count</th>
                        <th>Method</th>
                    </tr>
        """)
        
        for col, details in negative_fix_details.items():
            method_desc = {
                "clip_to_zero": "Replaced with 0",
                "cap_extreme_negative": "Capped extreme negative values",
                "no_change": "Identified but preserved (may be valid)"
            }
            f.write(f"""
                    <tr>
                        <td>{col}</td>
                        <td>{details['count']}</td>
                        <td>{method_desc.get(details['method'], details['method'])}</td>
                    </tr>
            """)
        
        # Finish HTML
        f.write(f"""
                </table>
                
                <div class="footer">
                    <p>Cleaning process completed on {CURRENT_DATE} by {CURRENT_USER}</p>
                    <p>Telecom Customer Churn Analysis Project</p>
                </div>
            </div>
        </body>
        </html>
        """)
    
    log_message("Cleaning complete!")
    log_message(f"Cleaned data saved to {output_file}")
    log_message(f"Summary report saved to {report_file}")
    log_message(f"Detailed HTML report saved to {html_report_file}")
    log_message(f"Distribution plots saved to {PLOTS_DIR}")

except Exception as e:
    log_message(f"Error during data cleaning: {str(e)}")
    import traceback
    log_message(traceback.format_exc())