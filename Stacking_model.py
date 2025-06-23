import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, f1_score, fbeta_score,
                           precision_score, recall_score, accuracy_score, make_scorer)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_and_prepare_data():
    df = pd.read_csv('Data.csv')
    exclude_cols = ['Client_ID', 'ARPU_Nov24', 'ARPU_Dec24', 'ARPU_Jan25']
    X = df.drop(columns=exclude_cols + ['Churn'])
    y = df['Churn']
    return X, y

def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred),
        'F2_Score': fbeta_score(y_true, y_pred, beta=2),
        'ROC_AUC': roc_auc_score(y_true, y_pred_proba)
    }

def create_visualizations(y_test, y_pred, y_pred_proba, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title(f'{model_name} - Confusion Matrix')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    axes[0,1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
    axes[0,1].plot([0, 1], [0, 1], 'k--')
    axes[0,1].set_title(f'{model_name} - ROC Curve')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].legend()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1,0].plot(recall, precision, linewidth=2)
    axes[1,0].set_title(f'{model_name} - Precision-Recall Curve')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    
    # Performance metrics bar plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'F2']
    metrics_values = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        fbeta_score(y_test, y_pred, beta=2)
    ]
    
    bars = axes[1,1].bar(metrics_names, metrics_values, color='red', alpha=0.7)
    axes[1,1].set_title(f'{model_name} - Performance Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("STACKING ENSEMBLE MODEL - TELECOM CHURN PREDICTION")
    print("="*60)
    
    # Load data
    X, y = load_and_prepare_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Convert to numpy arrays to avoid indexing issues
    X_train_balanced = np.array(X_train_balanced)
    y_train_balanced = np.array(y_train_balanced)
    
    print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
    print(f"After SMOTE - Target distribution: {np.bincount(y_train_balanced)}")
    
    # Define base models with LIGHTER parameters for speed
    print("Setting up base models...")
    
    # Logistic Regression
    lr_model = LogisticRegression(
        C=10, penalty='l2', 
        class_weight='balanced', random_state=42, max_iter=500
    )
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # XGBoost
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, 
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    
    # SVM (with calibration for probability estimates)
    svm_base = SVC(
        C=10, kernel='rbf', gamma='scale', 
        class_weight='balanced', random_state=42, probability=True
    )
    
    # Define base models list
    base_models = [
        ('logistic_regression', lr_model),
        ('random_forest', rf_model),
        ('xgboost', xgb_model),
        ('svm', svm_base)
    ]
    
    # Meta-classifier options to test (simplified)
    meta_classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    # Custom scorer for F2
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # Test different meta-classifiers (LIGHTER approach)
    best_stacking_score = 0
    best_stacking_model = None
    best_meta_name = None
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
    
    print("Testing different meta-classifiers...")
    
    for meta_name, meta_classifier in meta_classifiers.items():
        print(f"\nTesting meta-classifier: {meta_name}")
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            cv=2,  # Reduced CV for speed
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Use cross_val_score for simpler evaluation
        cv_scores = cross_val_score(stacking_model, X_train_balanced, y_train_balanced, 
                                   cv=cv, scoring=f2_scorer)
        
        mean_cv_score = np.mean(cv_scores)
        print(f"Mean CV F2 Score: {mean_cv_score:.4f} ± {np.std(cv_scores):.4f}")
        
        if mean_cv_score > best_stacking_score:
            best_stacking_score = mean_cv_score
            best_meta_name = meta_name
            best_stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_classifier,
                cv=2,
                stack_method='predict_proba',
                n_jobs=-1
            )
    
    print(f"\nBest meta-classifier: {best_meta_name}")
    print(f"Best CV F2 score: {best_stacking_score:.4f}")
    
    # Train the best stacking model on full training data
    print("\nTraining final stacking model...")
    best_stacking_model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions on test set
    y_pred = best_stacking_model.predict(X_test_scaled)
    y_pred_proba = best_stacking_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Get individual base model predictions for comparison
    print("\nEvaluating individual base models on test set:")
    base_model_results = {}
    
    for name, model in base_models:
        model.fit(X_train_balanced, y_train_balanced)
        base_pred = model.predict(X_test_scaled)
        base_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        base_metrics = calculate_metrics(y_test, base_pred, base_pred_proba)
        base_model_results[name] = base_metrics
        print(f"{name.replace('_', ' ').title()}: F2={base_metrics['F2_Score']:.4f}, "
              f"Precision={base_metrics['Precision']:.4f}, Recall={base_metrics['Recall']:.4f}")
    
    # Final cross-validation for stacking model
    print("\nPerforming final cross-validation...")
    final_cv_scores = cross_val_score(best_stacking_model, X_train_balanced, y_train_balanced, 
                                     cv=cv, scoring=f2_scorer)
    
    # Print results
    print("\n" + "="*60)
    print("STACKING ENSEMBLE RESULTS")
    print("="*60)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nCross-validation F2 scores: {final_cv_scores}")
    print(f"Mean CV F2 score: {np.mean(final_cv_scores):.4f} ± {np.std(final_cv_scores):.4f}")
    
    # Create visualizations
    create_visualizations(y_test, y_pred, y_pred_proba, "Stacking Ensemble")
    
    # FIXED: Feature importance from base models
    print("\nComputing ensemble feature importance...")
    ensemble_importance = np.zeros(X.shape[1])
    
    # Get feature importance from Random Forest (40% weight)
    rf_importance = best_stacking_model.named_estimators_['random_forest'].feature_importances_
    ensemble_importance += rf_importance * 0.4
    
    # Get feature importance from XGBoost (40% weight)
    xgb_importance = best_stacking_model.named_estimators_['xgboost'].feature_importances_
    ensemble_importance += xgb_importance * 0.4
    
    # Get feature importance from Logistic Regression (20% weight)
    lr_importance = np.abs(best_stacking_model.named_estimators_['logistic_regression'].coef_[0])
    lr_importance = lr_importance / np.sum(lr_importance)  # Normalize
    ensemble_importance += lr_importance * 0.2
    
    # NOTE: SVM doesn't have feature_importances_, so we skip it
    # The weights above add up to 100% (40% + 40% + 20% = 100%)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': ensemble_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Ensemble):")
    print(feature_importance.head(10))
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='red', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Ensemble Importance Score')
    plt.title('Stacking Ensemble - Feature Importance (Top 15)')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        plt.text(val + 0.001, i, f'{val:.3f}', va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('stacking_ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model comparison plot
    plt.figure(figsize=(12, 8))
    models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'Stacking Ensemble']
    f2_scores = [
        base_model_results['logistic_regression']['F2_Score'],
        base_model_results['random_forest']['F2_Score'],
        base_model_results['xgboost']['F2_Score'],
        base_model_results['svm']['F2_Score'],
        metrics['F2_Score']
    ]
    
    colors = ['skyblue', 'lightgreen', 'orange', 'pink', 'red']
    bars = plt.bar(models, f2_scores, color=colors, alpha=0.8)
    plt.title('Model Comparison - F2 Scores', fontsize=14, fontweight='bold')
    plt.ylabel('F2 Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, f2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_f2_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed report
    report_data = {
        'Model': 'Stacking Ensemble',
        'Meta_Classifier': best_meta_name,
        'Base_Models': [name for name, _ in base_models],
        'Test_Metrics': metrics,
        'CV_F2_Mean': np.mean(final_cv_scores),
        'CV_F2_Std': np.std(final_cv_scores),
        'Base_Model_Results': base_model_results,
        'Feature_Importance': feature_importance.to_dict('records')
    }
    
    # Save to CSV
    results_df = pd.DataFrame([{
        'Model': 'Stacking Ensemble',
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1_Score': metrics['F1_Score'],
        'F2_Score': metrics['F2_Score'],
        'ROC_AUC': metrics['ROC_AUC'],
        'CV_F2_Mean': np.mean(final_cv_scores),
        'CV_F2_Std': np.std(final_cv_scores)
    }])
    
    results_df.to_csv('stacking_ensemble_results.csv', index=False)
    
    # Save comprehensive comparison
    comparison_df = pd.DataFrame([
        {
            'Model': name.replace('_', ' ').title(),
            'F2_Score': results['F2_Score'],
            'Precision': results['Precision'],
            'Recall': results['Recall'],
            'ROC_AUC': results['ROC_AUC']
        }
        for name, results in base_model_results.items()
    ])
    
    # Add stacking ensemble results
    comparison_df = pd.concat([comparison_df, pd.DataFrame([{
        'Model': 'Stacking Ensemble',
        'F2_Score': metrics['F2_Score'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'ROC_AUC': metrics['ROC_AUC']
    }])], ignore_index=True)
    
    comparison_df.to_csv('all_models_comparison.csv', index=False)
    
    # Save classification report
    with open('stacking_ensemble_report.txt', 'w') as f:
        f.write("STACKING ENSEMBLE MODEL REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset shape: {X.shape}\n")
        f.write(f"Target distribution: {y.value_counts().to_dict()}\n")
        f.write(f"Churn rate: {y.mean():.3f}\n\n")
        f.write(f"Meta-classifier: {best_meta_name}\n")
        f.write(f"Base models: {[name for name, _ in base_models]}\n\n")
        f.write("BASE MODEL INDIVIDUAL PERFORMANCE:\n")
        for name, results in base_model_results.items():
            f.write(f"\n{name.replace('_', ' ').title()}:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nSTACKING ENSEMBLE PERFORMANCE:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"\nCross-validation F2 scores: {final_cv_scores.tolist()}\n")
        f.write(f"Mean CV F2 score: {np.mean(final_cv_scores):.4f} ± {np.std(final_cv_scores):.4f}\n\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nTOP 15 IMPORTANT FEATURES (Ensemble):\n")
        f.write("Feature importance computed from Random Forest (40%), XGBoost (40%), and Logistic Regression (20%).\n")
        f.write("Note: SVM feature importance excluded due to RBF kernel.\n")
        f.write(feature_importance.head(15).to_string())
        
        f.write("\n\nSTACKING METHODOLOGY:\n")
        f.write("1. Base models trained using 2-fold cross-validation for meta-features\n")
        f.write("2. Meta-classifier selected based on 3-fold cross-validation performance\n")
        f.write(f"3. Best meta-classifier: {best_meta_name}\n")
        f.write("4. Final model trained on full balanced dataset\n")
    
    print(f"\nResults saved to:")
    print("- stacking_ensemble_results.csv")
    print("- all_models_comparison.csv")
    print("- stacking_ensemble_report.txt")
    print("- stacking_ensemble_results.png")
    print("- stacking_ensemble_feature_importance.png")
    print("- model_comparison_f2_scores.png")
    
    print(f"\nStacking ensemble complete!")
    print(f"Best meta-classifier: {best_meta_name}")
    print(f"Final F2 Score: {metrics['F2_Score']:.4f}")

if __name__ == "__main__":
    main()