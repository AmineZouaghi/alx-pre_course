import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, f1_score, fbeta_score,
                           precision_score, recall_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load and prepare data
def load_and_prepare_data():
    df = pd.read_csv('Data.csv')
    
    # Exclude specified columns
    exclude_cols = ['Client_ID', 'ARPU_Nov24', 'ARPU_Dec24', 'ARPU_Jan25']
    
    # Separate features and target
    X = df.drop(columns=exclude_cols + ['Churn'])
    y = df['Churn']
    
    return X, y

# Evaluation metrics function
def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred),
        'F2_Score': fbeta_score(y_true, y_pred, beta=2),
        'ROC_AUC': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

# Visualization function
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
    axes[0,1].plot(fpr, tpr, linewidth=2)
    axes[0,1].plot([0, 1], [0, 1], 'k--')
    axes[0,1].set_title(f'{model_name} - ROC Curve')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1,0].plot(recall, precision, linewidth=2)
    axes[1,0].set_title(f'{model_name} - Precision-Recall Curve')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    
    # Feature Importance (for logistic regression, use coefficients)
    # This will be handled in main function
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("LOGISTIC REGRESSION MODEL - TELECOM CHURN PREDICTION")
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
    
    print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
    print(f"After SMOTE - Target distribution: {np.bincount(y_train_balanced)}")
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    }
    
    # Custom scorer for F2
    from sklearn.metrics import make_scorer
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid, cv=cv, scoring=f2_scorer, n_jobs=-1, verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F2 score: {grid_search.best_score_:.4f}")
    
    # Train best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Cross-validation scores
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train_balanced, y_train_balanced):
        X_cv_train, X_cv_val = X_train_balanced[train_idx], X_train_balanced[val_idx]
        y_cv_train, y_cv_val = y_train_balanced[train_idx], y_train_balanced[val_idx]
        
        best_model.fit(X_cv_train, y_cv_train)
        cv_pred = best_model.predict(X_cv_val)
        cv_scores.append(fbeta_score(y_cv_val, cv_pred, beta=2))
    
    # Print results
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*60)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nCross-validation F2 scores: {cv_scores}")
    print(f"Mean CV F2 score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': best_model.coef_[0],
        'abs_coefficient': np.abs(best_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Create visualizations
    create_visualizations(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='coefficient', y='feature')
    plt.title('Logistic Regression - Feature Coefficients (Top 15)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('logistic_regression_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed report
    report_data = {
        'Model': 'Logistic Regression',
        'Best_Parameters': grid_search.best_params_,
        'Test_Metrics': metrics,
        'CV_F2_Mean': np.mean(cv_scores),
        'CV_F2_Std': np.std(cv_scores),
        'Feature_Importance': feature_importance.to_dict('records')
    }
    
    # Save to CSV
    results_df = pd.DataFrame([{
        'Model': 'Logistic Regression',
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1_Score': metrics['F1_Score'],
        'F2_Score': metrics['F2_Score'],
        'ROC_AUC': metrics['ROC_AUC'],
        'CV_F2_Mean': np.mean(cv_scores),
        'CV_F2_Std': np.std(cv_scores)
    }])
    
    results_df.to_csv('logistic_regression_results.csv', index=False)
    
    # Save classification report
    with open('logistic_regression_report.txt', 'w') as f:
        f.write("LOGISTIC REGRESSION MODEL REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset shape: {X.shape}\n")
        f.write(f"Target distribution: {y.value_counts().to_dict()}\n")
        f.write(f"Churn rate: {y.mean():.3f}\n\n")
        f.write(f"Best parameters: {grid_search.best_params_}\n\n")
        f.write("TEST SET METRICS:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"\nCross-validation F2 scores: {cv_scores}\n")
        f.write(f"Mean CV F2 score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nTOP 15 IMPORTANT FEATURES:\n")
        f.write(feature_importance.head(15).to_string())
    
    print(f"\nResults saved to:")
    print("- logistic_regression_results.csv")
    print("- logistic_regression_report.txt")
    print("- logistic_regression_results.png")
    print("- logistic_regression_feature_importance.png")

if __name__ == "__main__":
    main()