"""
Stacking Ensemble model for telecom churn prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve
import time
import utils
from config import *

def train_stacking_model(X_train, y_train, preprocessor, base_models, meta_learner=None):
    """
    Train a Stacking Ensemble model combining multiple base models
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Target variable (churn)
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    base_models : list of tuples
        List of (name, model) tuples for base models
    meta_learner : estimator, optional
        Meta-model (defaults to LogisticRegression if None)
        
    Returns:
    --------
    pipeline : Pipeline
        Fitted stacking ensemble pipeline
    base_preds : dict
        Dictionary of base model predictions
    """
    print("\n" + "="*80)
    print("TRAINING STACKING ENSEMBLE MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Define meta-learner if not provided
    if meta_learner is None:
        meta_learner = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, 
                                         random_state=RANDOM_STATE, solver='liblinear')
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,  # 5-fold cross-validation for base models
        stack_method='predict_proba',
        n_jobs=-1,
        verbose=1
    )
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', stacking_clf)
    ])
    
    # Fit the stacking ensemble
    print(f"\nTraining stacking ensemble with {len(base_models)} base models...")
    pipeline.fit(X_train, y_train)
    
    # Generate predictions from individual models for visualization
    print("\nGenerating predictions from individual models for analysis...")
    base_preds = {}
    base_pipelines = {}
    
    for name, model in base_models:
        try:
            # Create pipeline for this base model
            model_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
            model_pipeline.fit(X_train, y_train)
            base_pipelines[name] = model_pipeline
            
            # Store predictions
            base_preds[name] = model_pipeline.predict_proba(X_train)[:, 1]
            print(f"✓ Generated predictions for {name}")
        except Exception as e:
            print(f"⚠️ Error getting predictions for {name}: {e}")
    
    # Add stacking ensemble predictions
    base_preds['stacking'] = pipeline.predict_proba(X_train)[:, 1]
    
    # Visualize model correlations
    visualize_model_correlations(base_preds, y_train)
    
    # Extract and visualize meta-model weights
    visualize_meta_model_weights(pipeline, base_models)
    
    # Visualize model performance comparison
    visualize_model_performance(base_preds, y_train)
    
    print(f"\n⏱️ Training completed in {time.time() - start_time:.2f} seconds")
    
    return pipeline, base_preds, base_pipelines

def visualize_model_correlations(base_preds, y_train):
    """Visualize correlations between model predictions"""
    try:
        # Create correlation matrix of model predictions
        pred_df = pd.DataFrame(base_preds)
        corr = pred_df.corr()
        
        # Add actual target for correlation with predictions
        pred_df['actual'] = y_train.values
        target_corr = pred_df.corr()['actual'].drop('actual').sort_values(ascending=False)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
        plt.title('Correlation Between Model Predictions', fontsize=14)
        plt.tight_layout()
        plt.savefig('Models/Results/stacking_model_correlations.png')
        plt.close()
        
        # Plot correlation with target
        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_corr.values, y=target_corr.index)
        plt.title('Model Correlation with Actual Churn', fontsize=14)
        plt.xlabel('Correlation with Target', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig('Models/Results/model_target_correlations.png')
        plt.close()
        
        print("\n📊 Model Correlations Analysis:")
        print("Lower correlation between models indicates better ensemble diversity")
        
        # Find most and least correlated model pairs
        corr_vals = corr.unstack()
        corr_vals = corr_vals[corr_vals < 0.99]  # Remove self-correlations
        most_corr = corr_vals.nlargest(1)
        least_corr = corr_vals.nsmallest(1)
        
        print(f"• Most correlated pair: {most_corr.index[0][0]} and {most_corr.index[0][1]} (r = {most_corr.values[0]:.4f})")
        print(f"• Least correlated pair: {least_corr.index[0][0]} and {least_corr.index[0][1]} (r = {least_corr.values[0]:.4f})")
        print(f"• Best individual model correlation with target: {target_corr.index[0]} (r = {target_corr.values[0]:.4f})")
        
    except Exception as e:
        print(f"⚠️ Error in correlation visualization: {e}")

def visualize_meta_model_weights(pipeline, base_models):
    """Extract and visualize meta-model weights"""
    try:
        meta_model = pipeline.named_steps['stacking'].final_estimator_
        if hasattr(meta_model, 'coef_'):
            meta_coefs = meta_model.coef_[0]
            meta_models = [name for name, _ in base_models]
            
            coef_df = pd.DataFrame({
                'Model': meta_models,
                'Weight': meta_coefs
            })
            coef_df = coef_df.sort_values('Weight', ascending=False)
            
            print("\n🔍 Meta-Learner Weights (Model Importance):")
            for index, row in coef_df.iterrows():
                print(f"• {row['Model']}: {row['Weight']:.4f}")
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Weight', y='Model', data=coef_df)
            plt.title('Meta-Learner Weights for Base Models', fontsize=14)
            plt.tight_layout()
            plt.savefig('Models/Results/meta_learner_weights.png')
            plt.close()
        else:
            print("⚠️ Meta-learner doesn't have accessible coefficients")
    except Exception as e:
        print(f"⚠️ Error extracting meta-model weights: {e}")

def visualize_model_performance(base_preds, y_train):
    """Visualize performance metrics for all models"""
    try:
        # Plot ROC curves for all models
        plt.figure(figsize=(10, 8))
        
        for name, preds in base_preds.items():
            fpr, tpr, _ = roc_curve(y_train, preds)
            plt.plot(fpr, tpr, label=f"{name}")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Base Models and Stacking Ensemble', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Models/Results/stacking_roc_curves.png')
        plt.close()
        
        # Plot Precision-Recall curves
        plt.figure(figsize=(10, 8))
        
        for name, preds in base_preds.items():
            precision, recall, _ = precision_recall_curve(y_train, preds)
            plt.plot(recall, precision, label=f"{name}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for Base Models and Stacking Ensemble', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Models/Results/stacking_precision_recall_curves.png')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Error in performance visualization: {e}")

def analyze_stacking_decisions(stacking_model, base_models_dict, X_test, feature_names=None):
    """
    Analyze how the stacking model makes decisions compared to base models
    
    Parameters:
    -----------
    stacking_model : Pipeline
        Fitted stacking ensemble pipeline
    base_models_dict : dict
        Dictionary of fitted base model pipelines
    X_test : DataFrame
        Test features
    feature_names : list, optional
        Feature names for visualization
        
    Returns:
    --------
    decision_df : DataFrame
        DataFrame with all predictions for analysis
    """
    print("\n" + "="*80)
    print("STACKING MODEL DECISION ANALYSIS")
    print("="*80)
    
    # Get predictions from all models
    all_preds = {}
    all_preds_binary = {}
    
    # Get stacking predictions
    stacking_proba = stacking_model.predict_proba(X_test)[:, 1]
    stacking_pred = (stacking_proba > 0.5).astype(int)
    all_preds['stacking'] = stacking_proba
    all_preds_binary['stacking'] = stacking_pred
    
    # Get base model predictions
    for name, model in base_models_dict.items():
        model_proba = model.predict_proba(X_test)[:, 1]
        model_pred = (model_proba > 0.5).astype(int)
        all_preds[name] = model_proba
        all_preds_binary[name] = model_pred
    
    # Create DataFrame with all predictions
    pred_df = pd.DataFrame(all_preds)
    pred_df_binary = pd.DataFrame(all_preds_binary)
    
    # Calculate model agreement
    pred_df['agreement_count'] = pred_df_binary.drop('stacking', axis=1).sum(axis=1)
    pred_df['agreement_ratio'] = pred_df['agreement_count'] / (len(base_models_dict))
    
    # Analyze where stacking disagrees with majority
    pred_df['majority_vote'] = (pred_df['agreement_count'] >= len(base_models_dict)/2).astype(int)
    pred_df['stacking_agrees_majority'] = (pred_df['stacking'] > 0.5) == (pred_df['majority_vote'] == 1)
    
    # Visualize agreement vs stacking probability
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='agreement_count', y='stacking', data=pred_df)
    plt.title('Stacking Prediction vs. Base Model Agreement', fontsize=14)
    plt.xlabel('Number of Base Models Predicting Churn', fontsize=12)
    plt.ylabel('Stacking Model Churn Probability', fontsize=12)
    plt.tight_layout()
    plt.savefig('Models/Results/stacking_agreement_analysis.png')
    plt.close()
    
    # Calculate how often stacking agrees with each base model
    agreement_rates = {}
    for name in base_models_dict.keys():
        agreement = (pred_df_binary['stacking'] == pred_df_binary[name]).mean() * 100
        agreement_rates[name] = agreement
    
    agreement_df = pd.DataFrame({
        'Model': list(agreement_rates.keys()),
        'Agreement with Stacking (%)': list(agreement_rates.values())
    }).sort_values('Agreement with Stacking (%)', ascending=False)
    
    print("\n🔍 Stacking Agreement Analysis:")
    print(f"• Stacking agrees with majority vote: {(pred_df['stacking_agrees_majority']).mean()*100:.2f}% of the time")
    
    for index, row in agreement_df.iterrows():
        print(f"• Agreement with {row['Model']}: {row['Agreement with Stacking (%)']:.2f}%")
    
    # Visualize agreement rates
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Agreement with Stacking (%)', y='Model', data=agreement_df)
    plt.title('Agreement Rates Between Stacking and Base Models', fontsize=14)
    plt.xlabel('Agreement Rate (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig('Models/Results/stacking_base_model_agreement.png')
    plt.close()
    
    return pred_df