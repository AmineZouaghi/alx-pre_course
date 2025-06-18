"""
Main script to run telecom churn prediction with stacking ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.model_selection import train_test_split
import utils
from logistic_regression_model import train_logistic_regression
from random_forest_model import train_random_forest
from xgboost_model import train_xgboost
from svm_model import train_svm
from stacking_model import train_stacking_model, analyze_stacking_decisions
from config import *

def main():
    """
    Main function to run the telecom churn prediction pipeline with stacking
    """
    print("="*80)
    print("TUNISIE TELECOM CHURN PREDICTION WITH STACKING ENSEMBLE")
    print("="*80)
    print(f"Current Date and Time: 2025-06-17 19:35:28")
    print(f"User: AmineZouaghi")
    print("="*80)
    
    # Start timer
    start_time = time.time()
    
    # Create output directories
    os.makedirs('Models/Results', exist_ok=True)
    os.makedirs('Models/Saved', exist_ok=True)
    
    # Load the data
    print("\nLoading data...")
    try:
        data_path = 'EDA/clean_data_for_modeling.csv'
        data = pd.read_csv(data_path)
        print(f"✓ Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Display data info
    print("\nData overview:")
    print(f"• Shape: {data.shape}")
    print(f"• Columns: {', '.join(data.columns[:5])}...")
    
    # Handle target variable
    target_col = 'Résiliation'  # Update this to match your actual target column
    if target_col in data.columns:
        print(f"✓ Target variable '{target_col}' found")
        target_distribution = data[target_col].value_counts(normalize=True) * 100
        print(f"• Target distribution: {target_distribution[1]:.2f}% churn, {target_distribution[0]:.2f}% non-churn")
    else:
        print(f"❌ Target variable '{target_col}' not found in data")
        potential_targets = [col for col in data.columns if 'churn' in col.lower() or 'résil' in col.lower()]
        if potential_targets:
            print(f"• Potential target columns: {', '.join(potential_targets)}")
        return
    
    # Prepare data for modeling
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nSplit data into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets")
    
    # Create preprocessor
    preprocessor = utils.create_preprocessor(X)
    
    # Initialize results list and model dictionary
    model_results = []
    base_models = []
    base_pipelines = {}
    
    # Train and evaluate base models
    print("\n" + "="*80)
    print("TRAINING BASE MODELS")
    print("="*80)
    
    feature_names = X_train.columns.tolist()
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, preprocessor, feature_names)
    print("Training Logistic Regression...")
    lr_results = utils.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    model_results.append(lr_results)
    base_models.append(('logistic', lr_model.named_steps['classifier']))
    base_pipelines['logistic'] = lr_model
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, preprocessor, feature_names)
    print("Training Random Forest...")
    if rf_model is None:
        print("❌ Random Forest model training failed. Skipping Random Forest.")
    else:
        print("✓ Random Forest model trained successfully")
    rf_results = utils.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    model_results.append(rf_results)
    base_models.append(('random_forest', rf_model.named_steps['classifier']))
    base_pipelines['random_forest'] = rf_model
    
    # Train SVM
    svm_model = train_svm(X_train, y_train, preprocessor, feature_names)
    print("Training SVM...")
    if svm_model is None:
        print("❌ SVM model training failed. Skipping SVM.")
    else:
        print("✓ SVM model trained successfully")
    svm_results = utils.evaluate_model(svm_model, X_test, y_test, "SVM")
    model_results.append(svm_results)
    base_models.append(('svm', svm_model.named_steps['classifier']))
    base_pipelines['svm'] = svm_model
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, preprocessor, feature_names)
    print("Training XGBoost...")
    if xgb_model is None:
        print("❌ XGBoost model training failed. Skipping XGBoost.")
    else:
        print("✓ XGBoost model trained successfully")
    xgb_results = utils.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    model_results.append(xgb_results)
    base_models.append(('xgboost', xgb_model.named_steps['classifier']))
    base_pipelines['xgboost'] = xgb_model
    
    # Train the stacking ensemble
    stacking_model, base_preds, _ = train_stacking_model(X_train, y_train, preprocessor, base_models, feature_names)
    print("Training Stacking Ensemble...")
    if stacking_model is None:
        print("❌ Stacking Ensemble model training failed. Exiting.")
        return
    else:
        print("✓ Stacking Ensemble model trained successfully")
    stacking_results = utils.evaluate_model(stacking_model, X_test, y_test, "Stacking Ensemble")
    model_results.append(stacking_results)
    
    # Compare all models
    comparison_df = utils.compare_models(model_results)
    
    # Save comparison results
    comparison_df.to_csv('Models/Results/model_comparison.csv')
    
    # Detailed analysis of stacking decisions
    decision_analysis = analyze_stacking_decisions(stacking_model, base_pipelines, X_test, feature_names)
    
    # Print final results and recommendations
    print("\n" + "="*80)
    print("FINAL RESULTS AND RECOMMENDATIONS")
    print("="*80)
    
    # Identify best model by F2 score
    best_model_name = comparison_df['f2'].idxmax()
    best_f2 = comparison_df.loc[best_model_name, 'f2']
    
    print(f"\n🏆 Best model by F2 score: {best_model_name} (F2 = {best_f2:.4f})")
    
    # Calculate stacking improvement
    stacking_f2 = comparison_df.loc['Stacking Ensemble', 'f2']
    best_base_model = comparison_df.drop('Stacking Ensemble').loc[:, 'f2'].idxmax()
    best_base_f2 = comparison_df.loc[best_base_model, 'f2']
    
    improvement = ((stacking_f2 - best_base_f2) / best_base_f2) * 100
    
    print(f"\n📈 Stacking Ensemble performance:")
    print(f"• F2 Score: {stacking_f2:.4f}")
    print(f"• Improvement over best base model ({best_base_model}): {improvement:.2f}%")
    
    # Business recommendations
    print("\n💼 Business Recommendations:")
    print("• Deploy the Stacking Ensemble model for churn prediction")
    print("• Focus retention efforts on customers with predicted churn probability > 0.6")
    print("• Monitor model performance monthly and retrain quarterly")
    print("• Collect additional customer satisfaction data to improve future models")
    
    # Print total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED")
    print("="*80)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("="*80)

if __name__ == "__main__":
    main()