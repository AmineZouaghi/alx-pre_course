import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick

# Suppress warnings
warnings.filterwarnings('ignore')

# Create EDA directory if it doesn't exist
if not os.path.exists('EDA'):
    os.makedirs('EDA')

# Set styles for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#3498DB", "#E74C3C"])

# Read the data
print("Loading data...")
df = pd.read_csv('Data Tunisie Telecom.csv', sep=',')
print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Save a copy of the original data
df_original = df.copy()

# -------------------------------------------------------------
# A. Data-quality profile
# -------------------------------------------------------------
def create_data_quality_profile(df):
    """Creates a comprehensive data quality profile table"""
    print("Creating data quality profile...")
    
    # Initialize stats dictionary
    stats_dict = {
        'Type': [],
        'Missing (%)': [],
        'Zeros (%)': [],
        'Min': [],
        'P25': [],
        'Median': [],
        'Mean': [],
        'P75': [],
        'Max': [],
        'Std': [],
        'Outliers (%)': []
    }
    
    for col in df.columns:
        # Data type
        stats_dict['Type'].append(df[col].dtype)
        
        # Missing values
        missing_pct = df[col].isna().mean() * 100
        stats_dict['Missing (%)'].append(round(missing_pct, 2))
        
        # Handling zeros (only for numeric columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            zeros_pct = (df[col] == 0).mean() * 100
            stats_dict['Zeros (%)'].append(round(zeros_pct, 2))
            
            # Basic statistics
            stats_dict['Min'].append(df[col].min())
            stats_dict['P25'].append(df[col].quantile(0.25))
            stats_dict['Median'].append(df[col].median())
            stats_dict['Mean'].append(df[col].mean())
            stats_dict['P75'].append(df[col].quantile(0.75))
            stats_dict['Max'].append(df[col].max())
            stats_dict['Std'].append(df[col].std())
            
            # Outlier detection (using IQR method)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_pct = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).mean() * 100
            stats_dict['Outliers (%)'].append(round(outlier_pct, 2))
        else:
            # For non-numeric columns, fill with N/A
            stats_dict['Zeros (%)'].append('N/A')
            stats_dict['Min'].append('N/A')
            stats_dict['P25'].append('N/A')
            stats_dict['Median'].append('N/A')
            stats_dict['Mean'].append('N/A')
            stats_dict['P75'].append('N/A')
            stats_dict['Max'].append('N/A')
            stats_dict['Std'].append('N/A')
            stats_dict['Outliers (%)'].append('N/A')
    
    # Create DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_dict, index=df.columns)
    
    # Save to CSV
    stats_df.to_csv('EDA/data_quality_profile.csv')
    
    # Create a formatted HTML version for better readability
    def color_cells(val, column_name):
        if isinstance(val, (int, float)):
            if 'Missing' in column_name and val > 30:
                return 'background-color: #FFCCCC'
            elif 'Outliers' in column_name and val > 10:
                return 'background-color: #FFFFCC'
        return ''
    
    styled_stats = stats_df.style.apply(lambda s: s.map(lambda x: color_cells(x, s.name)), axis=0)
    
    with open('EDA/data_quality_profile_styled.html', 'w') as f:
        f.write(styled_stats.to_html())
    
    print("Data quality profile saved to EDA folder")
    return stats_df

quality_profile = create_data_quality_profile(df)

# -------------------------------------------------------------
# B. Missing-value heat-map
# -------------------------------------------------------------
def create_missing_value_heatmap(df, sample_size=100):
    """Creates a heatmap visualization of missing values"""
    print("Creating missing value heatmap...")
    
    # Sample rows if dataset is large
    if df.shape[0] > sample_size:
        sampled_df = df.sample(sample_size, random_state=42)
    else:
        sampled_df = df
    
    # Create a binary matrix (True for missing, False for not missing)
    missing_matrix = sampled_df.isna()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(missing_matrix, cmap='viridis', cbar_kws={'label': 'Missing'})
    plt.title('Missing Value Patterns (Sample of Customers)', fontsize=16)
    plt.xlabel('Variables', fontsize=14)
    plt.ylabel('Customers (Sample)', fontsize=14)
    plt.tight_layout()
    plt.savefig('EDA/missing_value_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a summary heatmap showing percentage of missing values by column
    plt.figure(figsize=(12, 6))
    missing_percentage = df.isna().mean().sort_values(ascending=False) * 100
    
    # Only plot if there are missing values
    if missing_percentage.max() > 0:
        ax = sns.barplot(x=missing_percentage.index, y=missing_percentage.values, palette="viridis")
        plt.title('Percentage of Missing Values by Column', fontsize=16)
        plt.xlabel('Columns', fontsize=14)
        plt.ylabel('Missing (%)', fontsize=14)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha = 'center', va = 'bottom',
                           fontsize=10)
        
        plt.savefig('EDA/missing_percentage_by_column.png', dpi=300, bbox_inches='tight')
    else:
        plt.text(0.5, 0.5, "No missing values in the dataset", 
                 horizontalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig('EDA/missing_percentage_by_column.png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print("Missing value visualizations saved to EDA folder")

create_missing_value_heatmap(df)

# -------------------------------------------------------------
# C. Distribution plots
# -------------------------------------------------------------
def create_distribution_plots(df):
    """Creates distribution plots for each numeric column"""
    print("Creating distribution plots...")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a directory for distribution plots if it doesn't exist
    if not os.path.exists('EDA/distributions'):
        os.makedirs('EDA/distributions')
    
    # Determine grid size for combined plot
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    # Create individual plots
    for col in numeric_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine if log scale is appropriate
        use_log = False
        if ('ARPU' in col or 'Volume_Data' in col) and df[col].max() > 0:
            use_log = True
        
        # Histogram
        sns.histplot(df[col].dropna(), kde=True, ax=ax1, color=colors[0])
        ax1.set_title(f'Distribution of {col}')
        if use_log and df[col].min() > 0:
            ax1.set_xscale('log')
            ax1.set_title(f'Distribution of {col} (Log Scale)')
        
        # Box plot
        sns.boxplot(y=df[col].dropna(), ax=ax2, color=colors[2])
        ax2.set_title(f'Box Plot of {col}')
        if use_log and df[col].min() > 0:
            ax2.set_yscale('log')
            ax2.set_title(f'Box Plot of {col} (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(f'EDA/distributions/{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined plot for overall view
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=colors[i % len(colors)])
            axes[i].set_title(f'Distribution of {col}')
            
            # Use log scale for ARPU and data volume
            if ('ARPU' in col or 'Volume_Data' in col) and df[col].max() > 0 and df[col].min() > 0:
                axes[i].set_xscale('log')
                axes[i].set_title(f'Distribution of {col} (Log Scale)')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('EDA/all_distributions_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Distribution plots saved to EDA/distributions folder")

create_distribution_plots(df)

# -------------------------------------------------------------
# D. Correlation heat-map (Spearman)
# -------------------------------------------------------------
def create_correlation_heatmap(df):
    """Creates a Spearman correlation heatmap"""
    print("Creating correlation heatmap...")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate Spearman correlation
    corr_matrix = df[numeric_cols].corr(method='spearman').abs()
    
    # Create the heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='viridis', vmax=1, vmin=0, 
                annot=True, fmt='.2f', square=True, linewidths=.5)
    plt.title('Absolute Spearman Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('EDA/spearman_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the correlation matrix to CSV
    corr_matrix.to_csv('EDA/spearman_correlation_matrix.csv')
    
    # Identify and report highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                        corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        # Fix: Use utf-8 encoding when writing the file
        with open('EDA/high_correlation_pairs.txt', 'w', encoding='utf-8') as f:
            f.write("Highly correlated variable pairs (|ρ| > 0.7):\n")
            for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"{var1} ⟷ {var2}: {corr:.3f}\n")
    
    print("Correlation analysis saved to EDA folder")

create_correlation_heatmap(df)

# -------------------------------------------------------------
# E. Churn split visuals
# -------------------------------------------------------------
def create_churn_split_visuals(df):
    """Creates visualizations showing variable distributions split by churn status"""
    print("Creating churn split visualizations...")
    
    # Ensure the target variable is properly formatted
    df['Résiliation'] = df['Résiliation'].astype(int)
    
    # E1. Bar chart of churn rate by App_MyTT
    plt.figure(figsize=(10, 6))
    # Calculate churn rate by App_MyTT
    churn_by_app = df.groupby('App_MyTT')['Résiliation'].mean() * 100
    
    ax = sns.barplot(x=churn_by_app.index, y=churn_by_app.values, palette=[colors[2], colors[1]])
    plt.title('Churn Rate by MyTT App Installation Status', fontsize=16)
    plt.xlabel('MyTT App Installed (1 = Yes, 0 = No)', fontsize=14)
    plt.ylabel('Churn Rate (%)', fontsize=14)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   fontsize=12)
    
    plt.tight_layout()
    plt.savefig('EDA/churn_rate_by_app_installation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # E2. Key variables split by churn status
    key_variables = [
        # ARPU variables
        'ARPU_Nov24', 'ARPU_Dec24', 'ARPU_Jan25',
        # Minutes variables (one from each category)
        'Min_TT_Jan25', 'Min_HorsTT_Jan25', 'Min_TT_ServiceClient_Jan25',
        # Data volume
        'Volume_Data_Jan25',
        # Recharges
        'Nbr_Recharges_Jan25',
        # Tenure
        'Ancienneté_Abonné'
    ]
    
    # Create a directory for split visuals if it doesn't exist
    if not os.path.exists('EDA/churn_splits'):
        os.makedirs('EDA/churn_splits')
    
    # Create violin plots for each key variable
    for var in key_variables:
        if var in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Use violin plots with boxplots inside
            ax = sns.violinplot(x='Résiliation', y=var, data=df, palette=[colors[2], colors[1]], 
                               inner='box', cut=0)
            
            plt.title(f'Distribution of {var} by Churn Status', fontsize=16)
            plt.xlabel('Churn Status (1 = Churned, 0 = Retained)', fontsize=14)
            plt.ylabel(var, fontsize=14)
            
            # Use log scale for certain variables if they have positive values
            if ('ARPU' in var or 'Volume_Data' in var) and df[var].min() > 0:
                plt.yscale('log')
                plt.title(f'Distribution of {var} by Churn Status (Log Scale)', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(f'EDA/churn_splits/{var}_by_churn_status.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # E3. Create a combined visualization of key metrics averages by churn status
    plt.figure(figsize=(14, 8))
    
    # Calculate mean values for each variable by churn status
    mean_by_churn = df.groupby('Résiliation')[key_variables].mean()
    
    # Normalize the data for better visualization (divide each value by the maximum of that variable)
    norm_mean = mean_by_churn.div(mean_by_churn.max())
    
    # Transpose for better plotting
    norm_mean_t = norm_mean.T
    
    # Plot the normalized means
    ax = norm_mean_t.plot(kind='bar', color=[colors[2], colors[1]], figsize=(14, 8))
    plt.title('Normalized Average Values by Churn Status', fontsize=16)
    plt.xlabel('Variables', fontsize=14)
    plt.ylabel('Normalized Mean Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Retained (0)', 'Churned (1)'])
    plt.tight_layout()
    plt.savefig('EDA/normalized_means_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Churn split visualizations saved to EDA folder")

create_churn_split_visuals(df)

# -------------------------------------------------------------
# F. Trend features check
# -------------------------------------------------------------
def create_trend_features_visualization(df):
    """Creates visualizations of monthly trends split by churn status"""
    print("Creating trend features visualizations...")
    
    # Calculate month-over-month changes
    # ARPU changes
    df['ARPU_Change_Nov_Dec'] = ((df['ARPU_Dec24'] - df['ARPU_Nov24']) / df['ARPU_Nov24'].replace(0, np.nan)) * 100
    df['ARPU_Change_Dec_Jan'] = ((df['ARPU_Jan25'] - df['ARPU_Dec24']) / df['ARPU_Dec24'].replace(0, np.nan)) * 100
    
    # Minutes changes (on-net)
    df['Min_TT_Change_Nov_Dec'] = ((df['Min_TT_Dec24'] - df['Min_TT_Nov24']) / df['Min_TT_Nov24'].replace(0, np.nan)) * 100
    df['Min_TT_Change_Dec_Jan'] = ((df['Min_TT_Jan25'] - df['Min_TT_Dec24']) / df['Min_TT_Dec24'].replace(0, np.nan)) * 100
    
    # Minutes changes (off-net)
    df['Min_HorsTT_Change_Nov_Dec'] = ((df['Min_HorsTT_Dec24'] - df['Min_HorsTT_Nov24']) / df['Min_HorsTT_Nov24'].replace(0, np.nan)) * 100
    df['Min_HorsTT_Change_Dec_Jan'] = ((df['Min_HorsTT_Jan25'] - df['Min_HorsTT_Dec24']) / df['Min_HorsTT_Dec24'].replace(0, np.nan)) * 100
    
    # Data volume changes
    df['Data_Change_Nov_Dec'] = ((df['Volume_Data_Dec24'] - df['Volume_Data_Nov24']) / df['Volume_Data_Nov24'].replace(0, np.nan)) * 100
    df['Data_Change_Dec_Jan'] = ((df['Volume_Data_Jan25'] - df['Volume_Data_Dec24']) / df['Volume_Data_Dec24'].replace(0, np.nan)) * 100
    
    # Cap extreme percentage changes for better visualization
    for col in df.columns:
        if 'Change' in col:
            df[col] = df[col].clip(-100, 100)
    
    # Create a directory for trend visualizations
    if not os.path.exists('EDA/trends'):
        os.makedirs('EDA/trends')
    
    # Plot trends for key metrics
    trend_features = [
        ('ARPU_Nov24', 'ARPU_Dec24', 'ARPU_Jan25', 'ARPU'),
        ('Min_TT_Nov24', 'Min_TT_Dec24', 'Min_TT_Jan25', 'On-Net Minutes'),
        ('Min_HorsTT_Nov24', 'Min_HorsTT_Dec24', 'Min_HorsTT_Jan25', 'Off-Net Minutes'),
        ('Volume_Data_Nov24', 'Volume_Data_Dec24', 'Volume_Data_Jan25', 'Data Volume')
    ]
    
    # Create trend line plots
    for nov_col, dec_col, jan_col, label in trend_features:
        if all(col in df.columns for col in [nov_col, dec_col, jan_col]):
            plt.figure(figsize=(12, 7))
            
            # Prepare data for plotting
            churned = df[df['Résiliation'] == 1]
            retained = df[df['Résiliation'] == 0]
            
            # Calculate averages for each month
            churned_means = [churned[nov_col].mean(), churned[dec_col].mean(), churned[jan_col].mean()]
            retained_means = [retained[nov_col].mean(), retained[dec_col].mean(), retained[jan_col].mean()]
            
            # Plot lines
            plt.plot(['Nov', 'Dec', 'Jan'], churned_means, 'o-', color=colors[1], linewidth=3, label='Churned')
            plt.plot(['Nov', 'Dec', 'Jan'], retained_means, 'o-', color=colors[2], linewidth=3, label='Retained')
            
            plt.title(f'Average {label} Trend by Churn Status', fontsize=16)
            plt.xlabel('Month', fontsize=14)
            plt.ylabel(f'Average {label}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'EDA/trends/{label.replace(" ", "_").lower()}_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot percentage changes
    change_pairs = [
        ('ARPU_Change_Nov_Dec', 'ARPU_Change_Dec_Jan', 'ARPU'),
        ('Min_TT_Change_Nov_Dec', 'Min_TT_Change_Dec_Jan', 'On-Net Minutes'),
        ('Min_HorsTT_Change_Nov_Dec', 'Min_HorsTT_Change_Dec_Jan', 'Off-Net Minutes'),
        ('Data_Change_Nov_Dec', 'Data_Change_Dec_Jan', 'Data Volume')
    ]
    
    for nov_dec_change, dec_jan_change, label in change_pairs:
        plt.figure(figsize=(12, 7))
        
        # Create box plots for percentage changes
        change_data = pd.DataFrame({
            'Nov-Dec (Retained)': df[(df['Résiliation'] == 0)][nov_dec_change],
            'Dec-Jan (Retained)': df[(df['Résiliation'] == 0)][dec_jan_change],
            'Nov-Dec (Churned)': df[(df['Résiliation'] == 1)][nov_dec_change],
            'Dec-Jan (Churned)': df[(df['Résiliation'] == 1)][dec_jan_change]
        })
        
        sns.boxplot(data=change_data, palette=[colors[2], colors[2], colors[1], colors[1]])
        plt.title(f'{label} Percentage Change by Churn Status', fontsize=16)
        plt.ylabel('Percentage Change (%)', fontsize=14)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'EDA/trends/{label.replace(" ", "_").lower()}_percentage_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary of significant trend differences
    with open('EDA/trends/trend_summary.txt', 'w') as f:
        f.write("Summary of Trend Differences Between Churners and Non-Churners\n")
        f.write("=============================================================\n\n")
        
        for nov_dec_change, dec_jan_change, label in change_pairs:
            # Test for significant differences in Nov-Dec changes
            try:
                stat, pval = stats.mannwhitneyu(
                    df[df['Résiliation'] == 1][nov_dec_change].dropna(),
                    df[df['Résiliation'] == 0][nov_dec_change].dropna()
                )
                
                f.write(f"{label} - Nov to Dec Change:\n")
                f.write(f"  Churners median: {df[df['Résiliation'] == 1][nov_dec_change].median():.2f}%\n")
                f.write(f"  Non-churners median: {df[df['Résiliation'] == 0][nov_dec_change].median():.2f}%\n")
                f.write(f"  Mann-Whitney U p-value: {pval:.5f}\n")
                f.write(f"  Significant difference: {'Yes' if pval < 0.05 else 'No'}\n\n")
            except:
                f.write(f"{label} - Nov to Dec Change: Insufficient data for statistical test\n\n")
            
            # Test for significant differences in Dec-Jan changes
            try:
                stat, pval = stats.mannwhitneyu(
                    df[df['Résiliation'] == 1][dec_jan_change].dropna(),
                    df[df['Résiliation'] == 0][dec_jan_change].dropna()
                )
                
                f.write(f"{label} - Dec to Jan Change:\n")
                f.write(f"  Churners median: {df[df['Résiliation'] == 1][dec_jan_change].median():.2f}%\n")
                f.write(f"  Non-churners median: {df[df['Résiliation'] == 0][dec_jan_change].median():.2f}%\n")
                f.write(f"  Mann-Whitney U p-value: {pval:.5f}\n")
                f.write(f"  Significant difference: {'Yes' if pval < 0.05 else 'No'}\n\n")
            except:
                f.write(f"{label} - Dec to Jan Change: Insufficient data for statistical test\n\n")
    
    print("Trend feature visualizations saved to EDA/trends folder")

create_trend_features_visualization(df)

# -------------------------------------------------------------
# G. Tenure vs. churn curve
# -------------------------------------------------------------
def create_tenure_analysis(df):
    """Creates visualizations of churn rate across tenure buckets"""
    print("Creating tenure analysis...")
    
    # Create tenure buckets
    tenure_bins = [0, 180, 365, 730, float('inf')]
    tenure_labels = ['<6 months', '6-12 months', '12-24 months', '>24 months']
    
    df['Tenure_Bucket'] = pd.cut(df['Ancienneté_Abonné'], bins=tenure_bins, labels=tenure_labels)
    
    # Calculate churn rate by tenure bucket
    churn_by_tenure = df.groupby('Tenure_Bucket')['Résiliation'].agg(['mean', 'count'])
    churn_by_tenure['mean'] = churn_by_tenure['mean'] * 100  # Convert to percentage
    
    # Calculate 95% confidence intervals
    z = 1.96  # 95% confidence
    churn_by_tenure['ci_lower'] = churn_by_tenure['mean'] - z * np.sqrt(
        (churn_by_tenure['mean']/100 * (1 - churn_by_tenure['mean']/100)) / churn_by_tenure['count']
    ) * 100
    
    churn_by_tenure['ci_upper'] = churn_by_tenure['mean'] + z * np.sqrt(
        (churn_by_tenure['mean']/100 * (1 - churn_by_tenure['mean']/100)) / churn_by_tenure['count']
    ) * 100
    
    # Ensure confidence intervals are within [0, 100]
    churn_by_tenure['ci_lower'] = churn_by_tenure['ci_lower'].clip(0)
    churn_by_tenure['ci_upper'] = churn_by_tenure['ci_upper'].clip(upper=100)
    
    # Plot churn rate by tenure
    plt.figure(figsize=(12, 7))
    
    ax = sns.barplot(x=churn_by_tenure.index, y=churn_by_tenure['mean'], color=colors[0])
    
    # Add error bars for confidence intervals
    plt.errorbar(x=range(len(churn_by_tenure)), y=churn_by_tenure['mean'],
                yerr=(churn_by_tenure['mean']-churn_by_tenure['ci_lower'], 
                     churn_by_tenure['ci_upper']-churn_by_tenure['mean']),
                fmt='none', color='black', capsize=5)
    
    plt.title('Churn Rate by Tenure', fontsize=16)
    plt.xlabel('Tenure', fontsize=14)
    plt.ylabel('Churn Rate (%)', fontsize=14)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('EDA/churn_rate_by_tenure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the data to CSV
    churn_by_tenure.to_csv('EDA/churn_rate_by_tenure.csv')
    
    # Create a more detailed analysis with additional metrics
    tenure_analysis = df.groupby('Tenure_Bucket').agg({
        'Résiliation': 'mean',
        'ARPU_Jan25': 'mean',
        'Min_TT_Jan25': 'mean',
        'Min_HorsTT_Jan25': 'mean',
        'Volume_Data_Jan25': 'mean',
        'Ancienneté_Abonné': 'count'
    })
    
    # Rename columns for clarity
    tenure_analysis.columns = ['Churn_Rate', 'Avg_ARPU', 'Avg_On_Net_Minutes', 
                              'Avg_Off_Net_Minutes', 'Avg_Data_Volume', 'Customer_Count']
    
    # Convert churn rate to percentage
    tenure_analysis['Churn_Rate'] = tenure_analysis['Churn_Rate'] * 100
    
    # Calculate annual revenue at risk for each tenure bucket
    tenure_analysis['Avg_Annual_Revenue'] = tenure_analysis['Avg_ARPU'] * 12
    tenure_analysis['Revenue_At_Risk_Per_Customer'] = tenure_analysis['Avg_Annual_Revenue'] * tenure_analysis['Churn_Rate'] / 100
    tenure_analysis['Total_Annual_Revenue'] = tenure_analysis['Avg_Annual_Revenue'] * tenure_analysis['Customer_Count']
    tenure_analysis['Total_Revenue_At_Risk'] = tenure_analysis['Revenue_At_Risk_Per_Customer'] * tenure_analysis['Customer_Count']
    
    # Save detailed analysis to CSV
    tenure_analysis.to_csv('EDA/detailed_tenure_analysis.csv')
    
    print("Tenure analysis saved to EDA folder")

create_tenure_analysis(df)

# -------------------------------------------------------------
# H. Initial business findings slide
# -------------------------------------------------------------
def create_business_findings_summary(df):
    """Creates a text file with key business findings"""
    print("Creating business findings summary...")
    
    # Prepare data for findings
    # App impact on churn
    app_churn_rate = df.groupby('App_MyTT')['Résiliation'].mean()
    app_impact_ratio = app_churn_rate[0] / app_churn_rate[1] if app_churn_rate[1] > 0 else 0
    
    # Data usage drop analysis
    df['Data_Drop_Pct'] = ((df['Volume_Data_Dec24'] - df['Volume_Data_Jan25']) / 
                          df['Volume_Data_Dec24'].replace(0, np.nan)) * 100
    
    # Filter for customers with actual data usage in December
    data_users = df[df['Volume_Data_Dec24'] > 0]
    significant_drop = data_users[data_users['Data_Drop_Pct'] > 40]
    pct_churners_with_data_drop = (significant_drop[significant_drop['Résiliation'] == 1].shape[0] / 
                                  df[df['Résiliation'] == 1].shape[0]) * 100 if df[df['Résiliation'] == 1].shape[0] > 0 else 0
    
    # ARPU decline analysis
    df['ARPU_Drop_Pct'] = ((df['ARPU_Dec24'] - df['ARPU_Jan25']) / 
                          df['ARPU_Dec24'].replace(0, np.nan)) * 100
    significant_arpu_drop = df[df['ARPU_Drop_Pct'] > 30]
    pct_churners_with_arpu_drop = (significant_arpu_drop[significant_arpu_drop['Résiliation'] == 1].shape[0] / 
                                  df[df['Résiliation'] == 1].shape[0]) * 100 if df[df['Résiliation'] == 1].shape[0] > 0 else 0
    
    # Tenure analysis
    tenure_bins = [0, 180, 365, 730, float('inf')]
    tenure_labels = ['<6 months', '6-12 months', '12-24 months', '>24 months']
    df['Tenure_Bucket'] = pd.cut(df['Ancienneté_Abonné'], bins=tenure_bins, labels=tenure_labels)
    highest_churn_bucket = df.groupby('Tenure_Bucket')['Résiliation'].mean().idxmax()
    highest_churn_rate = df.groupby('Tenure_Bucket')['Résiliation'].mean().max() * 100
    
    # Customer service calls analysis
    service_calls = df[[col for col in df.columns if 'ServiceClient' in col]]
    df['Has_Service_Calls'] = (service_calls.sum(axis=1) > 0).astype(int)
    service_call_churn_rate = df.groupby('Has_Service_Calls')['Résiliation'].mean()
    service_call_impact = service_call_churn_rate[1] / service_call_churn_rate[0] if service_call_churn_rate[0] > 0 else 0
    
    # Write findings to file
    with open('EDA/key_business_findings.md', 'w') as f:
        f.write("# Key Business Findings from Initial EDA\n\n")
        
        # Finding 1: App impact
        f.write("## 1. MyTT App Adoption Impact\n")
        f.write(f"Customers without the MyTT app are **{app_impact_ratio:.1f}x more likely to churn** than those with the app. ")
        f.write(f"Churn rates: {app_churn_rate[0]*100:.1f}% (no app) vs. {app_churn_rate[1]*100:.1f}% (with app).\n\n")
        
        # Finding 2: Data usage drop
        f.write("## 2. Data Usage Decline as Warning Signal\n")
        f.write(f"**{pct_churners_with_data_drop:.1f}%** of churners showed a significant data usage drop (>40%) ")
        f.write("in the month before churning, suggesting disengagement precedes formal churn.\n\n")
        
        # Finding 3: ARPU decline
        f.write("## 3. Revenue Decline Pattern\n")
        f.write(f"**{pct_churners_with_arpu_drop:.1f}%** of churners experienced a substantial ARPU decline (>30%) ")
        f.write("in their final month, indicating reduced activity before departure.\n\n")
        
        # Finding 4: Tenure vulnerability
        f.write("## 4. Critical Tenure Window\n")
        f.write(f"The **{highest_churn_bucket}** tenure segment shows the highest churn rate at **{highest_churn_rate:.1f}%**, ")
        f.write("highlighting a critical vulnerability period in the customer lifecycle.\n\n")
        
        # Finding 5: Customer service calls
        f.write("## 5. Service Call Impact\n")
        f.write(f"Customers who contacted customer service are **{service_call_impact:.1f}x more likely to churn** ")
        f.write(f"({service_call_churn_rate[1]*100:.1f}% vs. {service_call_churn_rate[0]*100:.1f}%), ")
        f.write("suggesting service issues or unresolved problems drive departures.\n\n")
    
    # Create a more visual summary for presentation
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.95, "Key Business Findings from Initial EDA", 
             horizontalalignment='center', fontsize=20, fontweight='bold')
    
    findings = [
        f"MyTT App Impact: Non-app users {app_impact_ratio:.1f}x more likely to churn",
        f"Data Usage: {pct_churners_with_data_drop:.1f}% of churners showed >40% data drop before leaving",
        f"Revenue Pattern: {pct_churners_with_arpu_drop:.1f}% of churners had >30% ARPU decline in final month",
        f"Vulnerable Period: {highest_churn_bucket} segment has highest churn at {highest_churn_rate:.1f}%",
        f"Service Issues: Customers with service calls {service_call_impact:.1f}x more likely to churn"
    ]
    
    y_pos = 0.85
    for i, finding in enumerate(findings):
        plt.text(0.1, y_pos, f"{i+1}.", fontsize=16, fontweight='bold')
        plt.text(0.15, y_pos, finding, fontsize=14)
        y_pos -= 0.15
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('EDA/key_findings_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Business findings summary saved to EDA folder")

create_business_findings_summary(df)

# -------------------------------------------------------------
# I. Clean-data snapshot
# -------------------------------------------------------------
def create_clean_data_snapshot(df):
    """Creates a cleaned version of the dataset"""
    print("Creating clean data snapshot...")
    
    # Make a copy of the dataframe
    df_clean = df.copy()
    
    # 1. Handle missing values
    # For usage metrics, replace NaN with 0 (assuming missing means no usage)
    usage_cols = [col for col in df_clean.columns if any(x in col for x in ['Min_', 'Volume_Data_', 'Nbr_Recharges_'])]
    df_clean[usage_cols] = df_clean[usage_cols].fillna(0)
    
    # 2. Handle outliers - winsorize at 99th percentile
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        if col not in ['Résiliation', 'App_MyTT', 'Ancienneté_Abonné']:  # Skip binary and tenure columns
            p99 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(upper=p99)
    
    # 3. Create additional features for modeling based on EDA insights
    # Trend features - ARPU
    df_clean['ARPU_Trend_Nov_Jan'] = ((df_clean['ARPU_Jan25'] - df_clean['ARPU_Nov24']) / 
                                    df_clean['ARPU_Nov24'].replace(0, np.nan)) * 100
    df_clean['ARPU_Last_Change'] = ((df_clean['ARPU_Jan25'] - df_clean['ARPU_Dec24']) / 
                                  df_clean['ARPU_Dec24'].replace(0, np.nan)) * 100
    
    # Trend features - Minutes
    df_clean['Min_TT_Trend'] = ((df_clean['Min_TT_Jan25'] - df_clean['Min_TT_Nov24']) / 
                              df_clean['Min_TT_Nov24'].replace(0, np.nan)) * 100
    df_clean['Min_HorsTT_Trend'] = ((df_clean['Min_HorsTT_Jan25'] - df_clean['Min_HorsTT_Nov24']) / 
                                  df_clean['Min_HorsTT_Nov24'].replace(0, np.nan)) * 100
    
    # Trend features - Data
    df_clean['Data_Trend'] = ((df_clean['Volume_Data_Jan25'] - df_clean['Volume_Data_Nov24']) / 
                             df_clean['Volume_Data_Nov24'].replace(0, np.nan)) * 100
    
    # Tenure buckets
    tenure_bins = [0, 180, 365, 730, float('inf')]
    tenure_labels = ['<6 months', '6-12 months', '12-24 months', '>24 months']
    df_clean['Tenure_Bucket'] = pd.cut(df_clean['Ancienneté_Abonné'], bins=tenure_bins, labels=tenure_labels)
    
    # Ratio features
    df_clean['Off_Net_Ratio'] = df_clean['Min_HorsTT_Jan25'] / (df_clean['Min_TT_Jan25'] + df_clean['Min_HorsTT_Jan25']).replace(0, np.nan)
    
    # Service call indicator
    service_cols = [col for col in df_clean.columns if 'ServiceClient' in col]
    df_clean['Has_Service_Calls'] = (df_clean[service_cols].sum(axis=1) > 0).astype(int)
    
    # Fill NaN values in derived features with 0 for numerical stability
    derived_features = ['ARPU_Trend_Nov_Jan', 'ARPU_Last_Change', 'Min_TT_Trend', 
                       'Min_HorsTT_Trend', 'Data_Trend', 'Off_Net_Ratio']
    df_clean[derived_features] = df_clean[derived_features].fillna(0)
    
    # Cap extreme values in trend features
    trend_features = [col for col in df_clean.columns if 'Trend' in col or 'Change' in col]
    for col in trend_features:
        df_clean[col] = df_clean[col].clip(-100, 100)
    
    # Save the clean dataset
    df_clean.to_csv('EDA/clean_data_for_modeling.csv', index=False)
    
    # Create a data dictionary to document the features
    feature_descriptions = {
        # Original features
        'App_MyTT': 'Binary indicator of whether customer has MyTT app installed (1=Yes, 0=No)',
        'ARPU_Nov24': 'Average Revenue Per User for November 2024',
        'ARPU_Dec24': 'Average Revenue Per User for December 2024',
        'ARPU_Jan25': 'Average Revenue Per User for January 2025',
        'Min_TT_Nov24': 'On-net voice minutes for November 2024',
        'Min_TT_Dec24': 'On-net voice minutes for December 2024',
        'Min_TT_Jan25': 'On-net voice minutes for January 2025',
        'Min_TT_Fev25': 'On-net voice minutes for February 2025',
        'Min_HorsTT_Nov24': 'Off-net voice minutes for November 2024',
        'Min_HorsTT_Dec24': 'Off-net voice minutes for December 2024',
        'Min_HorsTT_Jan25': 'Off-net voice minutes for January 2025',
        'Min_TT_ServiceClient_Nov24': 'Minutes spent calling customer service in November 2024',
        'Min_TT_ServiceClient_Dec24': 'Minutes spent calling customer service in December 2024',
        'Min_TT_ServiceClient_Jan25': 'Minutes spent calling customer service in January 2025',
        'Nbr_Recharges_Nov24': 'Number of recharges in November 2024',
        'Nbr_Recharges_Dec24': 'Number of recharges in December 2024',
        'Nbr_Recharges_Jan25': 'Number of recharges in January 2025',
        'Volume_Data_Nov24': 'Data usage volume for November 2024',
        'Volume_Data_Dec24': 'Data usage volume for December 2024',
        'Volume_Data_Jan25': 'Data usage volume for January 2025',
        'Ancienneté_Abonné': 'Customer tenure in days',
        'Résiliation': 'Churn indicator (1=Churned, 0=Retained)',
        
        # Derived features
        'ARPU_Trend_Nov_Jan': 'Percentage change in ARPU from November to January',
        'ARPU_Last_Change': 'Percentage change in ARPU from December to January (last month)',
        'Min_TT_Trend': 'Percentage change in on-net minutes from November to January',
        'Min_HorsTT_Trend': 'Percentage change in off-net minutes from November to January',
        'Data_Trend': 'Percentage change in data volume from November to January',
        'Tenure_Bucket': 'Categorized tenure in groups: <6 months, 6-12 months, 12-24 months, >24 months',
        'Off_Net_Ratio': 'Ratio of off-net minutes to total minutes (higher values indicate more off-net usage)',
        'Has_Service_Calls': 'Binary indicator of whether customer called customer service in observation period'
    }
    
    # Write data dictionary to file
    with open('EDA/data_dictionary.md', 'w') as f:
        f.write("# Data Dictionary\n\n")
        f.write("## Original Features\n\n")
        
        for feature, description in feature_descriptions.items():
            if feature in df.columns:
                f.write(f"### {feature}\n")
                f.write(f"{description}\n\n")
        
        f.write("## Derived Features\n\n")
        
        for feature, description in feature_descriptions.items():
            if feature not in df.columns and feature in df_clean.columns:
                f.write(f"### {feature}\n")
                f.write(f"{description}\n\n")
    
    # Create a transformation log
    with open('EDA/data_transformation_log.md', 'w') as f:
        f.write("# Data Transformation Log\n\n")
        
        f.write("## Missing Value Treatment\n\n")
        f.write("* Usage metrics (minutes, data volume, recharges): Missing values replaced with 0\n")
        f.write("* Derived trend features: Missing values (due to division by zero) replaced with 0\n\n")
        
        f.write("## Outlier Treatment\n\n")
        f.write("* All usage and ARPU metrics: Winsorized at 99th percentile\n")
        f.write("* Trend and change features: Capped at ±100%\n\n")
        
        f.write("## Feature Engineering\n\n")
        f.write("* Trend features: Calculated percentage change over 3-month period\n")
        f.write("* Ratio features: Created off-net to total minutes ratio\n")
        f.write("* Categorical features: Created tenure buckets\n")
        f.write("* Binary indicators: Created service call flag\n\n")
        
        f.write("## Data Quality Summary\n\n")
        f.write(f"* Initial row count: {df.shape[0]}\n")
        f.write(f"* Final row count: {df_clean.shape[0]}\n")
        f.write(f"* Initial column count: {df.shape[1]}\n")
        f.write(f"* Final column count: {df_clean.shape[1]}\n")
    
    print("Clean data snapshot and documentation saved to EDA folder")

create_clean_data_snapshot(df)

def generate_data_cleaning_recommendations(df):
    """Generates a comprehensive report of data issues and cleaning recommendations"""
    print("Generating data cleaning recommendations...")
    
    # Create a report file
    with open('EDA/data_cleaning_recommendations.txt', 'w', encoding='utf-8') as f:
        f.write("=================================================================\n")
        f.write("            TUNISIE TELECOM DATA CLEANING REPORT                 \n")
        f.write("=================================================================\n")
        f.write(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        
        # -------------------- SECTION 1: MISSING VALUES --------------------
        f.write("1. MISSING VALUES ANALYSIS\n")
        f.write("-------------------------\n\n")
        
        # Count missing values per column
        missing_counts = df.isna().sum()
        missing_percent = (missing_counts / len(df)) * 100
        missing_info = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing Percent': missing_percent
        }).sort_values('Missing Percent', ascending=False)
        
        # Only show columns with missing values
        missing_cols = missing_info[missing_info['Missing Count'] > 0]
        
        if len(missing_cols) > 0:
            f.write(f"Found {len(missing_cols)} columns with missing values:\n\n")
            for col, row in missing_cols.iterrows():
                f.write(f"- {col}: {row['Missing Count']} values ({row['Missing Percent']:.2f}%)\n")
            
            # Group columns by missing value pattern
            f.write("\nMissing value patterns:\n")
            
            # Usage variables (minutes, data)
            usage_cols = [col for col in missing_cols.index if any(x in col for x in ['Min_', 'Volume_Data'])]
            if usage_cols:
                f.write("\n- Usage metrics (minutes, data): ")
                f.write("Missing values likely represent no usage. Recommend replacing with 0.\n")
            
            # ARPU variables
            arpu_cols = [col for col in missing_cols.index if 'ARPU' in col]
            if arpu_cols:
                f.write("\n- ARPU metrics: ")
                f.write("Missing values could indicate no billing data. ")
                f.write("Recommend replacing with 0 for active months, or using mean/median for analysis months.\n")
            
            # Structured recommendations
            f.write("\nRECOMMENDED ACTIONS:\n")
            f.write("1. Replace missing usage values (minutes, data) with 0\n")
            f.write("2. For ARPU, either replace with 0 or use median imputation based on tenure group\n")
            f.write("3. For any derived features, handle missing values after creation\n")
        else:
            f.write("No missing values found in the dataset.\n")
        
        # -------------------- SECTION 2: OUTLIERS --------------------
        f.write("\n\n2. OUTLIER ANALYSIS\n")
        f.write("------------------\n\n")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        outlier_stats = {}
        
        for col in numeric_cols:
            if col not in ['Résiliation', 'App_MyTT']:  # Skip binary variables
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers_lower = (df[col] < lower_bound).sum()
                outliers_upper = (df[col] > upper_bound).sum()
                total_outliers = outliers_lower + outliers_upper
                outlier_percent = (total_outliers / len(df)) * 100
                
                if total_outliers > 0:
                    outlier_stats[col] = {
                        'Total Outliers': total_outliers,
                        'Outlier Percent': outlier_percent,
                        'Lower Outliers': outliers_lower,
                        'Upper Outliers': outliers_upper,
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Median': df[col].median()
                    }
        
        # Sort by outlier percentage
        outlier_stats = {k: v for k, v in sorted(outlier_stats.items(), 
                                                key=lambda item: item[1]['Outlier Percent'], 
                                                reverse=True)}
        
        if outlier_stats:
            f.write(f"Found outliers in {len(outlier_stats)} columns:\n\n")
            
            for col, stats in outlier_stats.items():
                if stats['Outlier Percent'] > 5:  # Focus on significant outlier issues
                    f.write(f"- {col}: {stats['Total Outliers']} outliers ({stats['Outlier Percent']:.2f}%)\n")
                    f.write(f"  Range: [{stats['Min']} to {stats['Max']}], Median: {stats['Median']}\n")
                    if stats['Upper Outliers'] > stats['Lower Outliers']:
                        f.write(f"  Predominantly upper outliers, suggesting right-skewed distribution\n")
            
            # Group similar variables
            arpu_outliers = [col for col in outlier_stats.keys() if 'ARPU' in col]
            minutes_outliers = [col for col in outlier_stats.keys() if 'Min_' in col]
            data_outliers = [col for col in outlier_stats.keys() if 'Volume_Data' in col]
            
            f.write("\nOutlier patterns by variable group:\n")
            
            if arpu_outliers:
                f.write("\n- ARPU metrics: ")
                f.write("Right-skewed distribution with extremely high values. ")
                f.write("Likely represents premium customers rather than errors.\n")
            
            if minutes_outliers:
                f.write("\n- Call minutes: ")
                f.write("Right-skewed with extreme values. ")
                f.write("May represent business users or unusual calling patterns.\n")
            
            if data_outliers:
                f.write("\n- Data usage: ")
                f.write("Extreme right skew. ")
                f.write("Likely represents heavy data users rather than errors.\n")
            
            f.write("\nRECOMMENDED ACTIONS:\n")
            f.write("1. For modeling: Winsorize values at 99th percentile to limit extreme influence\n")
            f.write("2. For ARPU and usage metrics: Consider log transformation to normalize distributions\n")
            f.write("3. Flag extreme outliers (>99.9 percentile) for separate business analysis\n")
        else:
            f.write("No significant outliers detected in the dataset.\n")
        
        # -------------------- SECTION 3: CONSISTENCY ISSUES --------------------
        f.write("\n\n3. DATA CONSISTENCY ANALYSIS\n")
        f.write("---------------------------\n\n")
        
        # Check for zero values in key metrics
        zero_stats = {}
        key_metrics = [col for col in df.columns if any(x in col for x in ['ARPU_', 'Min_', 'Volume_Data_'])]
        
        for col in key_metrics:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                zero_percent = (zero_count / len(df)) * 100
                if zero_percent > 5:  # Only report significant zero patterns
                    zero_stats[col] = {
                        'Zero Count': zero_count,
                        'Zero Percent': zero_percent
                    }
        
        if zero_stats:
            f.write("Significant zero value patterns detected:\n\n")
            for col, stats in zero_stats.items():
                f.write(f"- {col}: {stats['Zero Count']} zeros ({stats['Zero Percent']:.2f}%)\n")
            
            f.write("\nZero values may indicate:")
            f.write("\n- Inactive customers or services")
            f.write("\n- Data collection issues")
            f.write("\n- Service suspensions")
            
            f.write("\n\nRECOMMENDED ACTIONS:")
            f.write("\n1. Create binary flags for 'has_usage' alongside continuous metrics")
            f.write("\n2. For trend features, handle divisions by zero appropriately")
            f.write("\n3. Consider filtering inactive customers for certain analyses\n")
        else:
            f.write("No significant patterns of zero values detected.\n")
        
        # Check for negative values (which might be errors)
        negative_cols = {}
        for col in numeric_cols:
            if col in df.columns and not col.startswith('Change_'):  # Exclude change metrics which can be negative
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    neg_percent = (neg_count / len(df)) * 100
                    negative_cols[col] = {
                        'Negative Count': neg_count,
                        'Negative Percent': neg_percent
                    }
        
        if negative_cols:
            f.write("\nNegative values detected in columns that should be non-negative:\n\n")
            for col, stats in negative_cols.items():
                f.write(f"- {col}: {stats['Negative Count']} negative values ({stats['Negative Percent']:.2f}%)\n")
            
            f.write("\nRECOMMENDED ACTIONS:")
            f.write("\n1. Replace negative ARPU/usage values with 0 (likely billing adjustments)")
            f.write("\n2. Flag accounts with negative values for manual review\n")
        else:
            f.write("\nNo negative values detected in metrics that should be non-negative.\n")
        
        # -------------------- SECTION 4: FEATURE ENGINEERING --------------------
        f.write("\n\n4. RECOMMENDED FEATURE ENGINEERING\n")
        f.write("----------------------------------\n\n")
        
        f.write("Based on the data analysis, the following feature engineering is recommended:\n\n")
        
        f.write("A. Trend Features:\n")
        f.write("   - Month-over-month percentage changes for ARPU, minutes, and data\n")
        f.write("   - Three-month trend directions (increasing, stable, decreasing)\n")
        f.write("   - Volatility metrics (standard deviation of month-to-month changes)\n\n")
        
        f.write("B. Ratio Features:\n")
        f.write("   - Off-net to on-net call ratio\n")
        f.write("   - Customer service call ratio to total calls\n")
        f.write("   - Data to ARPU ratio (data intensity)\n\n")
        
        f.write("C. Categorical Features:\n")
        f.write("   - Tenure buckets (<6 months, 6-12 months, 1-2 years, >2 years)\n")
        f.write("   - Usage pattern segments (high/medium/low for voice and data)\n")
        f.write("   - ARPU tier categories\n\n")
        
        f.write("D. Binary Indicators:\n")
        f.write("   - Service call flag (contacted customer service: yes/no)\n")
        f.write("   - Significant drop flags (>30% drop in usage or ARPU)\n")
        f.write("   - Zero usage flags\n\n")
        
        # -------------------- SECTION 5: FINAL CLEANING PLAN --------------------
        f.write("\n\n5. COMPREHENSIVE DATA CLEANING PLAN\n")
        f.write("----------------------------------\n\n")
        
        f.write("STEP 1: Handle Missing Values\n")
        f.write("- Replace missing usage values with 0\n")
        f.write("- Impute missing ARPU with 0 or median by tenure group\n")
        f.write("- Create missing value indicators for sensitive analyses\n\n")
        
        f.write("STEP 2: Address Outliers\n")
        f.write("- Winsorize extreme values at 99th percentile\n")
        f.write("- Apply log transformation to heavily skewed variables\n")
        f.write("- Flag extreme outliers for separate analysis\n\n")
        
        f.write("STEP 3: Fix Inconsistencies\n")
        f.write("- Replace negative values with 0 for usage metrics\n")
        f.write("- Create binary indicators for zero usage\n")
        f.write("- Handle potential duplicates or errors\n\n")
        
        f.write("STEP 4: Create Derived Features\n")
        f.write("- Calculate trend and change features\n")
        f.write("- Create ratio features\n")
        f.write("- Generate categorical features\n")
        f.write("- Add binary indicators\n\n")
        
        f.write("STEP 5: Prepare for Modeling\n")
        f.write("- Scale continuous features\n")
        f.write("- Encode categorical variables\n")
        f.write("- Create train/validation/test splits\n")
        f.write("- Generate final modeling dataset\n\n")
        
        f.write("=================================================================\n")
        f.write("                      END OF REPORT                              \n")
        f.write("=================================================================\n")
    
    print("Data cleaning recommendations saved to EDA/data_cleaning_recommendations.txt")
    return True

# Add this to the end of your main script
generate_data_cleaning_recommendations(df)

# -------------------------------------------------------------
# Final Summary
# -------------------------------------------------------------
def create_eda_summary():
    """Creates a summary of all EDA outputs"""
    print("Creating EDA summary...")
    
    with open('EDA/eda_summary.md', 'w') as f:
        f.write("# Tunisie Télécom Churn Prediction EDA Summary\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document summarizes the exploratory data analysis (EDA) conducted on the Tunisie Télécom ")
        f.write("customer dataset for churn prediction. The analysis explored 22 original variables including ")
        f.write("engagement indicators, revenue metrics, usage patterns, and customer tenure.\n\n")
        
        f.write("## Key Outputs\n\n")
        
        f.write("### A. Data Quality Profile\n")
        f.write("* Comprehensive statistics for all variables\n")
        f.write("* Identification of missing values and outliers\n")
        f.write("* Data type verification and constraints\n\n")
        
        f.write("### B. Missing Value Analysis\n")
        f.write("* Visual pattern detection of missing data\n")
        f.write("* Quantification of missing values by variable\n\n")
        
        f.write("### C. Distribution Analysis\n")
        f.write("* Histograms and box plots for all numeric variables\n")
        f.write("* Log-scale analysis for heavy-tailed distributions (ARPU, data volume)\n\n")
        
        f.write("### D. Correlation Analysis\n")
        f.write("* Spearman correlation heatmap\n")
        f.write("* Identification of multicollinearity concerns\n\n")
        
        f.write("### E. Churn Split Analysis\n")
        f.write("* Comparison of variable distributions between churners and non-churners\n")
        f.write("* Identification of significant behavioral differences\n\n")
        
        f.write("### F. Trend Analysis\n")
        f.write("* Month-over-month changes in key metrics\n")
        f.write("* Identification of pattern differences preceding churn\n\n")
        
        f.write("### G. Tenure Analysis\n")
        f.write("* Churn rate across customer lifecycle stages\n")
        f.write("* Identification of high-risk tenure windows\n\n")
        
        f.write("### H. Business Findings\n")
        f.write("* Initial insights from exploratory analysis\n")
        f.write("* Quantification of key behavioral patterns\n\n")
        
        f.write("### I. Clean Data\n")
        f.write("* Processed dataset ready for modeling\n")
        f.write("* Feature engineering based on EDA insights\n")
        f.write("* Documentation of transformations\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Feature engineering based on identified patterns\n")
        f.write("2. Model development using logistic regression and XGBoost\n")
        f.write("3. Model evaluation with AUC and business lift metrics\n")
        f.write("4. Deployment of prediction system and dashboard\n")
    
    print("EDA summary created")

create_eda_summary()

print("EDA completed! All outputs saved to the EDA folder.")