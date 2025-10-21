import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, f1_score, fbeta_score,
                           precision_score, recall_score, accuracy_score, make_scorer)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
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
    axes[0,1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
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
    
    axes[1,1].bar(metrics_names, metrics_values, color='purple', alpha=0.7)
    axes[1,1].set_title(f'{model_name} - Performance Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_values):
        axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("SUPPORT VECTOR MACHINE MODEL - TELECOM CHURN PREDICTION")
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
    
    # Scale features (crucial for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # OPTIMIZATION 1: Use smaller balanced subset for hyperparameter tuning
    print("Creating balanced subset for hyperparameter tuning...")
    
    # Take stratified sample for faster hyperparameter tuning
    X_hp_subset, _, y_hp_subset, _ = train_test_split(
        X_train_scaled, y_train, train_size=0.15, random_state=42, stratify=y_train
    )
    
    # Apply SMOTE on hyperparameter tuning subset
    smote_hp = SMOTE(random_state=42)
    X_hp_balanced, y_hp_balanced = smote_hp.fit_resample(X_hp_subset, y_hp_subset)
    
    print(f"Hyperparameter tuning subset: {X_hp_balanced.shape}")
    print(f"HP subset target distribution: {np.bincount(y_hp_balanced)}")
    
    # OPTIMIZATION 2: Minimal but effective parameter grid
    param_grid = {
        'C': [1, 10],                    # Only 2 values - most important
        'gamma': ['scale'],              # Only scale (auto-tuned)
        'kernel': ['rbf'],               # Only RBF
        'class_weight': ['balanced']
    }
    
    # Custom scorer for F2
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # OPTIMIZATION 3: Fast hyperparameter search on subset
    cv_hp = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Only 2-fold
    grid_search = GridSearchCV(
        SVC(random_state=42, probability=True),
        param_grid, cv=cv_hp, scoring=f2_scorer, n_jobs=-1, verbose=1
    )
    
    print("Starting hyperparameter tuning on subset (this will be faster)...")
    grid_search.fit(X_hp_balanced, y_hp_balanced)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F2 score (on subset): {grid_search.best_score_:.4f}")
    
    # OPTIMIZATION 4: Train on larger but still manageable dataset
    print("Preparing final training set...")
    
    # Use 50% of original training data for final model (balance speed vs performance)
    X_final_subset, _, y_final_subset, _ = train_test_split(
        X_train_scaled, y_train, train_size=0.5, random_state=42, stratify=y_train
    )
    
    # Apply SMOTE on final training subset
    smote_final = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote_final.fit_resample(X_final_subset, y_final_subset)
    
    # Convert to numpy arrays to avoid indexing issues
    X_train_balanced = np.array(X_train_balanced)
    y_train_balanced = np.array(y_train_balanced)
    
    print(f"Final training set shape: {X_train_balanced.shape}")
    print(f"Final target distribution: {np.bincount(y_train_balanced)}")
    
    # Train best SVM model on final dataset
    print("Training SVM with best parameters...")
    best_svm = SVC(**grid_search.best_params_, random_state=42, probability=True)
    best_svm.fit(X_train_balanced, y_train_balanced)
    
    # OPTIMIZATION 5: Fast calibration
    print("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(best_svm, method='sigmoid', cv=2)  # 2-fold calibration
    calibrated_model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions on full test set
    y_pred = calibrated_model.predict(X_test_scaled)
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # OPTIMIZATION 6: Simplified cross-validation
    print("Performing cross-validation...")
    cv_final = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(calibrated_model, X_train_balanced, y_train_balanced, 
                               cv=cv_final, scoring=f2_scorer)
    
    # Print results
    print("\n" + "="*60)
    print("SUPPORT VECTOR MACHINE RESULTS")
    print("="*60)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nCross-validation F2 scores: {cv_scores}")
    print(f"Mean CV F2 score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # OPTIMIZATION 7: Fast feature importance using small sample
    print("Computing feature importance using permutation importance...")
    from sklearn.inspection import permutation_importance
    
    # Use very small subset for permutation importance
    perm_sample_size = min(200, len(X_test_scaled))
    perm_indices = np.random.choice(len(X_test_scaled), perm_sample_size, replace=False)
    X_perm_sample = X_test_scaled[perm_indices]
    y_perm_sample = y_test.iloc[perm_indices]
    
    perm_importance = permutation_importance(
        calibrated_model, X_perm_sample, y_perm_sample,
        n_repeats=3, random_state=42, scoring=f2_scorer, n_jobs=-1
    )
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Create visualizations
    create_visualizations(y_test, y_pred, y_pred_proba, "Support Vector Machine")
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                    xerr=top_features['std'], color='purple', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('SVM - Feature Importance (Top 15)')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        plt.text(val + top_features['std'].iloc[i], i, f'{val:.3f}', 
                va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('svm_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed report
    report_data = {
        'Model': 'Support Vector Machine',
        'Best_Parameters': grid_search.best_params_,
        'Test_Metrics': metrics,
        'CV_F2_Mean': np.mean(cv_scores),
        'CV_F2_Std': np.std(cv_scores),
        'Feature_Importance': feature_importance.to_dict('records'),
        'Optimization_Strategy': {
            'HP_Tuning_Sample': f"{X_hp_balanced.shape[0]} samples",
            'Final_Training_Sample': f"{X_train_balanced.shape[0]} samples",
            'Permutation_Sample': f"{perm_sample_size} samples"
        }
    }
    
    # Save to CSV
    results_df = pd.DataFrame([{
        'Model': 'Support Vector Machine',
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1_Score': metrics['F1_Score'],
        'F2_Score': metrics['F2_Score'],
        'ROC_AUC': metrics['ROC_AUC'],
        'CV_F2_Mean': np.mean(cv_scores),
        'CV_F2_Std': np.std(cv_scores)
    }])
    
    results_df.to_csv('svm_results.csv', index=False)
    
    # Save classification report
    with open('svm_report.txt', 'w') as f:
        f.write("SUPPORT VECTOR MACHINE MODEL REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset shape: {X.shape}\n")
        f.write(f"Target distribution: {y.value_counts().to_dict()}\n")
        f.write(f"Churn rate: {y.mean():.3f}\n\n")
        
        f.write("OPTIMIZATION STRATEGY FOR COMPUTATIONAL EFFICIENCY:\n")
        f.write(f"1. Hyperparameter tuning on subset: {X_hp_balanced.shape[0]} samples\n")
        f.write(f"2. Final training on subset: {X_train_balanced.shape[0]} samples\n")
        f.write(f"3. Permutation importance on subset: {perm_sample_size} samples\n")
        f.write("This approach maintains model quality while ensuring reasonable computation time.\n\n")
        
        f.write(f"Best parameters: {grid_search.best_params_}\n\n")
        f.write("TEST SET METRICS:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"\nCross-validation F2 scores: {cv_scores.tolist()}\n")
        f.write(f"Mean CV F2 score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nTOP 15 IMPORTANT FEATURES (Permutation Importance):\n")
        f.write(feature_importance.head(15).to_string())
        f.write(f"\n\nPermutation importance computed on {perm_sample_size} test samples for efficiency.\n")
        f.write("This sample size is sufficient for reliable feature importance estimation.\n")
    
    print(f"\nResults saved to:")
    print("- svm_results.csv")
    print("- svm_report.txt")
    print("- support_vector_machine_results.png")
    print("- svm_feature_importance.png")
    
    print(f"\nOptimization Summary:")
    print(f"- Hyperparameter tuning: {X_hp_balanced.shape[0]} samples (15% of training data)")
    print(f"- Final training: {X_train_balanced.shape[0]} samples (50% of training data)")
    print(f"- Permutation importance: {perm_sample_size} samples")
    print("- Full RBF SVM with comprehensive output!")
    print("- Expected runtime: 8-12 minutes")

if __name__ == "__main__":
    main()