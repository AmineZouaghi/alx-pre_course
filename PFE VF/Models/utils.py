"""
Utility functions for churn prediction models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (confusion_matrix, classification_report, fbeta_score, 
                            precision_score, recall_score, f1_score, roc_auc_score,
                            make_scorer)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os
import time
from config import *

def load_and_preprocess_data(file_path="EDA/clean_data_for_modeling.csv", test_size=0.2, random_state=42):
    """
    Load data and perform initial preprocessing
    
    Parameters:
    -----------
    file_path : str, default="EDA/clean_data_for_modeling.csv"
        Path to the data file
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target
    y_test : Series
        Test target
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    all_features : list
        List of all feature names
    feature_types : dict
        Dictionary of feature types
    original_features : list
        List of original feature names
    """
    print("Loading and preprocessing data...")
    
    # Load data
    try:
        data = pd.read_csv(file_path)
        print(f"✓ Successfully loaded data from {file_path}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None, None, None, None, None
    
    # Print data info
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Identify target column (assuming it's 'Résiliation' or contains 'churn')
    target_col = None
    for col in data.columns:
        if col.lower() in ['résiliation', 'resiliation', 'churn']:
            target_col = col
            break
    
    if target_col is None:
        print("❌ Could not identify target column.")
        return None, None, None, None, None, None, None, None
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    original_features = X.columns
    
    # Identify categorical and numerical features
    cat_features = []
    num_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 10:
            cat_features.append(col)
        else:
            num_features.append(col)
    
    print(f"\nFeature Types:")
    print(f"• Numerical Features: {len(num_features)}")
    print(f"• Categorical Features: {len(cat_features)}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='passthrough'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Print class distribution
    print("\nClass distribution:")
    train_counts = pd.Series(y_train).value_counts(normalize=True) * 100
    test_counts = pd.Series(y_test).value_counts(normalize=True) * 100
    
    for cls in sorted(y.unique()):
        print(f"• Class {cls}: {train_counts.get(cls, 0):.2f}% in training, {test_counts.get(cls, 0):.2f}% in test")
    
    # Return all required data
    all_features = num_features + cat_features
    feature_types = {'numerical': num_features, 'categorical': cat_features}
    
    return X_train, X_test, y_train, y_test, preprocessor, all_features, feature_types, original_features

def f2_score(y_true, y_pred):
    """
    Calculate F2 score - emphasizes recall over precision
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
        
    Returns:
    --------
    score : float
        F2 score
    """
    return fbeta_score(y_true, y_pred, beta=2)

def create_f2_scorer():
    """
    Create a scorer function for F2 score that emphasizes recall over precision
    
    Returns:
    --------
    scorer : callable
        A scorer function that can be used with GridSearchCV or RandomizedSearchCV
    """
    return make_scorer(fbeta_score, beta=2)

def handle_imbalance(pipeline, method='class_weight', step_name='classifier'):
    """
    Modify the pipeline to handle class imbalance
    
    Parameters:
    -----------
    pipeline : Pipeline
        The sklearn Pipeline object
    method : str, default='class_weight'
        Method to handle imbalance: 'class_weight', 'smote', or 'none'
    step_name : str, default='classifier'
        The name of the classifier step in the pipeline
        
    Returns:
    --------
    pipeline : Pipeline
        Modified pipeline with imbalance handling
    """
    if method == 'none':
        return pipeline
    
    if method == 'class_weight':
        # Get classifier from pipeline
        classifier = pipeline.named_steps[step_name]
        
        # Set class_weight to 'balanced' if the classifier supports it
        if hasattr(classifier, 'class_weight'):
            classifier.set_params(class_weight='balanced')
            print("✓ Set class_weight='balanced' to handle imbalance")
        else:
            print("⚠️ Classifier doesn't support class_weight parameter")
        
        return pipeline
    
    elif method == 'smote':
        # Extract steps from the original pipeline
        steps = [(name, step) for name, step in pipeline.steps]
        
        # Find the position of the classifier
        classifier_idx = next((i for i, (name, _) in enumerate(steps) if name == step_name), None)
        
        if classifier_idx is None:
            print(f"⚠️ Step '{step_name}' not found in pipeline")
            return pipeline
        
        # Insert SMOTE before the classifier
        steps.insert(classifier_idx, ('smote', SMOTE(random_state=42)))
        
        # Create a new pipeline with imbalanced-learn
        return ImbPipeline(steps)
    
    else:
        print(f"⚠️ Unknown imbalance handling method: {method}")
        return pipeline

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and print results
    
    Parameters:
    -----------
    model : estimator
        Fitted model
    X_test : DataFrame
        Test features
    y_test : Series
        Test target
    model_name : str
        Name of the model
        
    Returns:
    --------
    results : dict
        Dictionary with model name, metrics, and predictions
    """
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except:
        print("⚠️ Model doesn't support probability predictions")
    
    # Calculate metrics
    f2 = f2_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC AUC if probabilities are available
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"\n📊 {model_name} Performance Metrics:")
    print(f"• F2 Score: {f2:.4f}")
    print(f"• Precision: {precision:.4f}")
    print(f"• Recall: {recall:.4f}")
    print(f"• F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"• ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save plot
    plt.savefig(f'results/{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.close()
    
    # Return results
    return {
        'model_name': model_name,
        'f2': f2,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def compare_models(model_results):
    """
    Compare multiple models based on their metrics
    
    Parameters:
    -----------
    model_results : list
        List of dictionaries with model evaluation results
        
    Returns:
    --------
    comparison_df : DataFrame
        DataFrame with model comparison
    """
    print("\nComparing models...")
    
    # Create DataFrame from results
    results_data = []
    for result in model_results:
        model_data = {
            'Model': result['model_name'],
            'F2 Score': result['f2'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1']
        }
        
        if result['roc_auc'] is not None:
            model_data['ROC AUC'] = result['roc_auc']
            
        results_data.append(model_data)
    
    results_df = pd.DataFrame(results_data)
    results_df.set_index('Model', inplace=True)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    results_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save plot
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    # Print best model for each metric
    print("\n🏆 Best Models:")
    for metric in results_df.columns:
        best_model = results_df[metric].idxmax()
        best_score = results_df[metric].max()
        print(f"• Best {metric}: {best_model} ({best_score:.4f})")
    
    return results_df