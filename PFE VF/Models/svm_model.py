"""SVM model with advanced optimization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import time
import utils
from config import *

def train_svm(X_train, y_train, preprocessor, feature_names):
    """
    Train an SVM model with advanced hyperparameter tuning
    """
    print("\n" + "="*80)
    print("TRAINING SVM MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Create base pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=RANDOM_STATE, probability=True))
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
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            class_weight_param: class_weight_values,
            'classifier__degree': [2, 3, 4]  # Only used by poly kernel
        }
        
        # Create F2 scorer
        f2_scorer = utils.create_f2_scorer()
        
        # Set up cross-validation strategy
        cv = utils.get_cv_strategy(y_train)
        
        # Set up GridSearchCV
        search = GridSearchCV(
            pipeline,
            param_grid,
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
            'classifier__C': np.logspace(-2, 3, 100),
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 20)),
            class_weight_param: class_weight_values,
            'classifier__degree': [2, 3, 4, 5]  # Only used by poly kernel
        }
        
        # Create F2 scorer
        f2_scorer = utils.create_f2_scorer()
        
        # Set up cross-validation strategy
        cv = utils.get_cv_strategy(y_train)
        
        # Set up RandomizedSearchCV
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
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
            Real(1e-2, 1e3, prior='log-uniform', name='classifier__C'),
            Categorical(['linear', 'rbf', 'sigmoid'], name='classifier__kernel'),
            Real(1e-4, 1, prior='log-uniform', name='classifier__gamma'),
            Categorical(class_weight_values, name=class_weight_param)
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
    print(f"⚠️ SVM training may take longer than other models, especially with large datasets.")
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
            
            # Group by kernel for better visualization
            for kernel in cv_results['param_classifier__kernel'].unique():
                kernel_mask = cv_results['param_classifier__kernel'] == kernel
                if not any(kernel_mask):
                    continue
                    
                plt.semilogx(cv_results.loc[kernel_mask, 'param_classifier__C'], 
                             cv_results.loc[kernel_mask, 'mean_test_score'], 
                             marker='o', linestyle='-', label=f'Test Score ({kernel})')
            
            plt.xlabel('C Parameter (log scale)', fontsize=12)
            plt.ylabel('F2 Score', fontsize=12)
            plt.title('Cross-Validation Scores vs. C Parameter by Kernel', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Plot heatmap for gamma vs C if rbf kernel was used
            best_kernel = search.best_params_.get('classifier__kernel')
            if best_kernel in ['rbf', 'poly', 'sigmoid'] and 'param_classifier__gamma' in cv_results.columns:
                # Filter for best kernel
                kernel_mask = cv_results['param_classifier__kernel'] == best_kernel
                if any(kernel_mask):
                    kernel_results = cv_results[kernel_mask]
                    
                    # Try to create a pivot table for C vs gamma
                    try:
                        pivot_data = pd.pivot_table(
                            kernel_results, 
                            values='mean_test_score',
                            index='param_classifier__gamma', 
                            columns='param_classifier__C'
                        )
                        
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f')
                        plt.title(f'F2 Score Heatmap: Gamma vs C ({best_kernel} kernel)', fontsize=14)
                        plt.xlabel('C Parameter', fontsize=12)
                        plt.ylabel('Gamma Parameter', fontsize=12)
                        plt.tight_layout()
                        plt.show()
                    except:
                        print("⚠️ Could not create gamma vs C heatmap")
    except:
        print("⚠️ Could not plot cross-validation results")
    
    print(f"\n⏱️ Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the model
    model_filename = utils.save_model(best_model, "SVM", search.best_params_)
    
    return best_model