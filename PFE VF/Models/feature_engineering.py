"""Advanced feature engineering and selection techniques"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (RFE, RFECV, SelectKBest, f_classif, 
                                      mutual_info_classif, VarianceThreshold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import spearmanr
from config import *
import time

def analyze_features(X, y, feature_names):
    """
    Perform exploratory feature analysis
    """
    print("="*80)
    print("FEATURE ANALYSIS")
    print("="*80)
    
    # Calculate correlations with target
    correlations = {}
    for i, col in enumerate(feature_names):
        try:
            # Try to calculate Spearman correlation
            corr, _ = spearmanr(X[:, i], y)
            correlations[col] = corr
        except:
            correlations[col] = 0
    
    # Sort correlations
    sorted_correlations = {k: v for k, v in sorted(correlations.items(), 
                                                 key=lambda item: abs(item[1]), 
                                                 reverse=True)}
    
    # Print top correlations
    print("\n🔍 Top Correlated Features with Target:")
    for i, (col, corr) in enumerate(sorted_correlations.items()):
        if i < 15:  # Show top 15
            print(f"• {col}: {corr:.4f}")
    
    # Plot top correlations
    plt.figure(figsize=(12, 8))
    top_features = list(sorted_correlations.keys())[:15]
    top_corrs = [sorted_correlations[feature] for feature in top_features]
    sns.barplot(x=top_corrs, y=top_features)
    plt.title('Top Features by Correlation with Target', fontsize=14)
    plt.xlabel('Correlation (Spearman)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Return useful information for feature selection
    return sorted_correlations

def select_features(X_train, y_train, feature_names, method='rfe', n_features=TOP_N_FEATURES):
    """
    Select the most important features using various methods
    """
    print("\n" + "="*80)
    print(f"FEATURE SELECTION USING {method.upper()}")
    print("="*80)
    
    start_time = time.time()
    selected_features = []
    
    if method == 'rfe':
        # Recursive Feature Elimination
        print("Performing Recursive Feature Elimination...")
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
        rfe.fit(X_train, y_train)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_features = [feature for i, feature in enumerate(feature_names) if selected_mask[i]]
        
        # Get feature ranking
        feature_ranking = pd.DataFrame({
            'Feature': feature_names,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        # Plot feature ranking
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Ranking', y='Feature', data=feature_ranking.head(n_features))
        plt.title('Feature Ranking from RFE', fontsize=14)
        plt.xlabel('Ranking (lower is better)', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    elif method == 'kbest':
        # SelectKBest with mutual information
        print("Performing SelectKBest with mutual information...")
        selector = SelectKBest(mutual_info_classif, k=n_features)
        selector.fit(X_train, y_train)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [feature for i, feature in enumerate(feature_names) if selected_mask[i]]
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': feature_names,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        # Plot feature scores
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Score', y='Feature', data=feature_scores.head(n_features))
        plt.title('Feature Scores from SelectKBest', fontsize=14)
        plt.xlabel('Score (higher is better)', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    elif method == 'variance':
        # Variance Threshold
        print("Performing Variance Threshold...")
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(X_train)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [feature for i, feature in enumerate(feature_names) if selected_mask[i]]
        
        # Get feature variances
        feature_variances = pd.DataFrame({
            'Feature': feature_names,
            'Variance': selector.variances_
        }).sort_values('Variance', ascending=False)
        
        # Plot feature variances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Variance', y='Feature', data=feature_variances.head(n_features))
        plt.title('Feature Variances', fontsize=14)
        plt.xlabel('Variance (higher has more information)', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    elif method == 'pca':
        # Principal Component Analysis
        print("Performing PCA...")
        pca = PCA(n_components=n_features)
        pca.fit(X_train)
        
        # Explain variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Plot explained variance
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, 
                label='Individual Explained Variance')
        plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
                label='Cumulative Explained Variance')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        plt.title('Explained Variance by Components', fontsize=14)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Explained Variance', fontsize=12)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
        # For PCA, we don't select specific features but return the number of components
        selected_features = list(range(n_features))  # Just return the number of components
    
    else:
        print(f"❌ Unknown feature selection method: {method}")
        return feature_names
    
    print(f"\n✅ Selected {len(selected_features)} features in {time.time() - start_time:.2f} seconds")
    if len(selected_features) <= 20:
        print("Selected features:")
        for feature in selected_features:
            print(f"• {feature}")
    
    return selected_features