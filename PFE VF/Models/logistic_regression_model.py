"""Logistic Regression model with advanced optimization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import time
import utils
from config import *

def train_logistic_regression(X_train, y_train, preprocessor, feature_names):
    """
    Train a Logistic Regression model with advanced hyperparameter tuning
    """
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Create base pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=5000))
    ])
    
    # Apply class imbalance handling if needed
    if HANDLE_IMBALANCE and IMBALANCE_METHOD == 'class_weight':
        class_weight_param = 'classifier__class_weight'
        class_weight_values = ['balanced']
    else:
        class_weight_param = 'classifier__class_weight'
        class_weight_values = [None]
    
    # Handle class imbalance with SMOTE if configured
    pipeline = utils.handle_imbalance(pipeline)
    
    # Set up parameter search based on optimization method
    if OPTIMIZATION_METHOD == 'grid':
        print("Using Grid Search for hyperparameter optimization...")
        
        # Define hyperparameters to tune
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
            'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            class_weight_param: class_weight_values
        }
        
        # Filter out invalid combinations
        param_combinations = []
        for penalty in param_grid['classifier__penalty']:
            for solver in param_grid['classifier__solver']:
                # Handle invalid combinations
                if penalty == 'l1' and solver in ['newton-cg', 'sag']:
                    continue
                if penalty == 'elasticnet' and solver != 'saga':
                    continue
                if penalty is None and solver in ['liblinear']:
                    continue
                
                for C in param_grid['classifier__C']:
                    for class_weight in param_grid[class_weight_param]:
                        param_combinations.append({
                            'classifier__C': C,
                            'classifier__penalty': penalty,
                            'classifier__solver': solver,
                            class_weight_param: class_weight
                        })
        
        # Create F2 scorer
        f2_scorer = utils.create_f2_scorer()
        
        # Set up cross-validation strategy
        cv = utils.get_cv_strategy(y_train)
        
        # Set up GridSearchCV
        search = GridSearchCV(
            pipeline,
            param_combinations,
            cv=cv,
            scoring=f2_scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
    
    elif OPTIMIZATION_METHOD == 'random':
        print("Using Random Search for hyperparameter optimization...")
        
        # Define hyperparameters to tune
        param_distributions = {
            'classifier__C': np.logspace(-3, 3, 100),
            'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
            'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            class_weight_param: class_weight_values
        }
        
        # Filter out invalid combinations
        param_combinations = []
        for penalty in param_distributions['classifier__penalty']:
            for solver in param_distributions['classifier__solver']:
                # Handle invalid combinations
                if penalty == 'l1' and solver in ['newton-cg', 'sag']:
                    continue
                if penalty == 'elasticnet' and solver != 'saga':
                    continue
                if penalty is None and solver in ['liblinear']:
                    continue
                
                param_combinations.append({
                    'classifier__penalty': [penalty],
                    'classifier__solver': [solver],
                    class_weight_param: param_distributions[class_weight_param]
                })
                
                if len(param_combinations) > 0:
                    param_combinations[-1]['classifier__C'] = param_distributions['classifier__C']
        
        # Create F2 scorer
        f2_scorer = utils.create_f2_scorer()
        
        # Set up cross-validation strategy
        cv = utils.get_cv_strategy(y_train)
        
        # Pick the first valid combination for RandomizedSearchCV
        # (it doesn't support filtering invalid combinations directly)
        search = RandomizedSearchCV(
            pipeline,
            param_combinations[0],
            n_iter=N_ITER,
            cv=cv,
            scoring=f2_scorer,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            return_train_score=True
        )
    
    elif OPTIMIZATION_METHOD == 'bayesian':
        print("Using Bayesian Optimization for hyperparameter tuning...")
        
        # Define search space
        search_space = [
            Real(1e-3, 1e3, prior='log-uniform', name='classifier__C'),
            Categorical(['l2', 'none'], name='classifier__penalty'),
            Categorical(['newton-cg', 'lbfgs', 'sag'], name='classifier__solver'),
            Categorical(class_weight_values, name='classifier__class_weight')
        ]
        
        # Create F2 scorer
        f2_scorer = utils.create_f2_scorer()
        
        # Set up cross-validation strategy
        cv = utils.get_cv_strategy(y_train)
        
        # Set up BayesSearchCV
        search = BayesSearchCV(
            pipeline,
            search_space,
            n_iter=N_ITER,
            cv=cv,
            scoring=f2_scorer,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            return_train_score=True
        )
    
    else:
        raise ValueError(f"Unknown optimization method: {OPTIMIZATION_METHOD}")
    
    # Fit the search
    print(f"\nStarting hyperparameter optimization using {CV_FOLDS}-fold cross-validation...")
    search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("\n🔍 Best parameters:")
    for param, value in search.best_params_.items():
        print(f"• {param}: {value}")
    
    print(f"\n📊 Best cross-validation F2 score: {search.best_score_:.4f}")
    
    # Get the best estimator
    best_model = search.best_estimator_
    
    # Get cross-validation results
    try:
        cv_results = pd.DataFrame(search.cv_results_)
        
        # Plot cross-validation results for C parameter
        if 'param_classifier__C' in cv_results.columns:
            plt.figure(figsize=(10, 6))
            plt.semilogx(cv_results['param_classifier__C'], 
                        cv_results['mean_test_score'], 
                        marker='o', linestyle='-', label='Test Score')
            plt.semilogx(cv_results['param_classifier__C'], 
                        cv_results['mean_train_score'], 
                        marker='x', linestyle='--', label='Train Score')
            plt.xlabel('C Parameter (log scale)', fontsize=12)
            plt.ylabel('F2 Score', fontsize=12)
            plt.title('Cross-Validation Scores vs. C Parameter', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    except:
        print("⚠️ Could not plot cross-validation results")
    
    print(f"\n⏱️ Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the model
    model_filename = utils.save_model(best_model, "LogisticRegression", search.best_params_)
    
    return best_model