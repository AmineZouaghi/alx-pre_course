# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Data.csv')

# Basic information
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
else:
    print("\nNo missing values found in the dataset.")

# Identify target variable
if 'Churn' in df.columns:
    target_col = 'Churn'
else:
    # Assume the last column is the target
    target_col = df.columns[-1]

# Check target distribution
churn_distribution = df[target_col].value_counts(normalize=True) * 100
print(f"\nTarget Distribution (%):\n{churn_distribution}")
print(f"\nChurn Rate: {churn_distribution[1]:.2f}%")

# Drop identifier and ARPU columns as specified
columns_to_drop = ['Client_ID', 'ARPU_Nov24', 'ARPU_Dec24', 'ARPU_Jan25']
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(f"\nRemoved columns: {[col for col in columns_to_drop if col in df.columns]}")
print(f"Remaining features: {df_cleaned.shape[1] - 1}")

# Statistical summary of numerical features
numerical_stats = df_cleaned.describe().T
numerical_stats['range'] = numerical_stats['max'] - numerical_stats['min']
numerical_stats['coefficient_of_variation'] = (numerical_stats['std'] / numerical_stats['mean']).abs()
print("\nStatistical Summary of Top 5 Numerical Features:")
print(numerical_stats.sort_values(by='coefficient_of_variation', ascending=False).head())

# Visualizations
print("\nGenerating visualizations...")

# 1. Target Distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=target_col, data=df)
plt.title('Distribution of Churn', fontsize=16)
plt.xlabel('Churn (1 = Yes, 0 = No)', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add percentage labels
total = len(df)
for p in ax.patches:
    height = p.get_height()
    percentage = height / total * 100
    ax.text(p.get_x() + p.get_width()/2., height + 5, 
            f'{height} ({percentage:.1f}%)', 
            ha="center", fontsize=12)

plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap of Top Features
plt.figure(figsize=(14, 12))
correlation_matrix = df_cleaned.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Features', fontsize=16)
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Top correlations with target
corr_with_target = correlation_matrix[target_col].sort_values(ascending=False)
top_pos_corr = corr_with_target[corr_with_target > 0][1:11]  # Excluding target itself
top_neg_corr = corr_with_target[corr_with_target < 0][:10]

plt.figure(figsize=(12, 8))
sns.barplot(x=top_pos_corr.values, y=top_pos_corr.index)
plt.title('Top 10 Positive Correlations with Churn', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.savefig('top_positive_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x=top_neg_corr.values, y=top_neg_corr.index)
plt.title('Top 10 Negative Correlations with Churn', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.savefig('top_negative_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Distribution of engineered features by churn status
engineered_features = [
    'Average_Call_Duration', 'Revenue_Growth', 'Peak_to_OffPeak_Ratio', 
    'Service_Call_Ratio', 'Recharge_Consistency'
]

available_engineered_features = [f for f in engineered_features if f in df_cleaned.columns]

if available_engineered_features:
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(available_engineered_features):
        plt.subplot(3, 2, i+1)
        sns.boxplot(x=target_col, y=feature, data=df_cleaned)
        plt.title(f'{feature} by Churn Status', fontsize=14)
        plt.xlabel('Churn (1 = Yes, 0 = No)', fontsize=12)
        plt.tight_layout()
    plt.savefig('engineered_features_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Monthly usage patterns
monthly_usage_cols = [col for col in df_cleaned.columns if any(month in col for month in ['Nov24', 'Dec24', 'Jan25'])]
usage_by_month = {}

# Group by month
for col in monthly_usage_cols:
    if 'Nov24' in col:
        month = 'November'
    elif 'Dec24' in col:
        month = 'December'
    elif 'Jan25' in col:
        month = 'January'
    else:
        continue
    
    if month not in usage_by_month:
        usage_by_month[month] = []
    
    usage_by_month[month].append(col)

# Plot time series for key metrics by churn status
if monthly_usage_cols:
    # Find common metrics across months
    metrics = set()
    for col in monthly_usage_cols:
        base_name = col.split('_')[0] + '_' + col.split('_')[1]
        if 'Nov24' in col or 'Dec24' in col or 'Jan25' in col:
            metrics.add(base_name.replace('_Nov24', '').replace('_Dec24', '').replace('_Jan25', ''))
    
    for metric in metrics:
        nov_col = next((col for col in monthly_usage_cols if metric in col and 'Nov24' in col), None)
        dec_col = next((col for col in monthly_usage_cols if metric in col and 'Dec24' in col), None)
        jan_col = next((col for col in monthly_usage_cols if metric in col and 'Jan25' in col), None)
        
        if nov_col and dec_col and jan_col:
            plt.figure(figsize=(12, 6))
            
            # For churned customers
            churned = df_cleaned[df_cleaned[target_col] == 1]
            plt.plot(['Nov', 'Dec', 'Jan'], 
                    [churned[nov_col].mean(), churned[dec_col].mean(), churned[jan_col].mean()], 
                    'r-o', linewidth=3, label='Churned Customers')
            
            # For non-churned customers
            not_churned = df_cleaned[df_cleaned[target_col] == 0]
            plt.plot(['Nov', 'Dec', 'Jan'], 
                    [not_churned[nov_col].mean(), not_churned[dec_col].mean(), not_churned[jan_col].mean()], 
                    'g-o', linewidth=3, label='Retained Customers')
            
            plt.title(f'Average {metric.replace("_", " ")} Trend by Churn Status', fontsize=16)
            plt.ylabel('Average Value', fontsize=14)
            plt.xlabel('Month', fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{metric}_trend.png', dpi=300, bbox_inches='tight')
            plt.close()

# 6. App Usage Impact on Churn
if 'App_MyTT' in df_cleaned.columns:
    plt.figure(figsize=(10, 6))
    app_churn = pd.crosstab(df_cleaned['App_MyTT'], df_cleaned[target_col], normalize='index') * 100
    app_churn.plot(kind='bar', stacked=True)
    plt.title('Impact of App Usage on Churn Rate', fontsize=16)
    plt.xlabel('App Usage (1 = Yes, 0 = No)', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(['Retained', 'Churned'])
    
    # Add percentage labels
    for i, p in enumerate(plt.gca().patches):
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 5:  # Only add labels if segment is large enough
            plt.gca().text(x+width/2, y+height/2, f'{height:.1f}%', 
                    ha='center', va='center', fontsize=12)
    
    plt.savefig('app_usage_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Feature Distribution Analysis
numerical_features = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
if target_col in numerical_features:
    numerical_features.remove(target_col)

top_features = abs(correlation_matrix[target_col][numerical_features]).sort_values(ascending=False).head(6).index.tolist()

plt.figure(figsize=(18, 15))
for i, feature in enumerate(top_features):
    plt.subplot(3, 2, i+1)
    
    # Plot distribution by churn status
    sns.histplot(data=df_cleaned, x=feature, hue=target_col, kde=True, element="step", bins=30)
    plt.title(f'Distribution of {feature} by Churn Status', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(['Retained', 'Churned'])
    
plt.tight_layout()
plt.savefig('top_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Outlier Analysis
plt.figure(figsize=(16, 12))
for i, feature in enumerate(top_features):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x=feature, data=df_cleaned)
    plt.title(f'Boxplot of {feature}', fontsize=14)
    plt.tight_layout()
plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nEDA completed! Visualizations saved as PNG files.")
print("="*50)