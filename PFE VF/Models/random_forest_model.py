"""Random Forest model with advanced optimization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import time
import utils
from config import *

def train_random_forest(X_train, y_train, preprocessor, feature_names):
    """
    Train a Random Forest model with advanced hyperparameter tuning
    """
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Create base pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    # Apply class imbalance handling if needed
    if HANDLE_IMBALANCE and IMBALANCE_METHOD == 'class_weight':
        class_weight_param = 'classifier__class_weight'
        class_weight_values = ['balanced', 'balanced_subsample']
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
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 15, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None],
            class_weight_param: class_weight_values
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
            'classifier__n_estimators': np.arange(50, 500, 10),
            'classifier__max_depth': [None] + list(np.arange(5, 50, 5)),
            'classifier__min_samples_split': np.arange(2, 20, 1),
            'classifier__min_samples_leaf': np.arange(1, 10, 1),
            'classifier__max_features': ['sqrt', 'log2', None],
            class_weight_param: class_weight_values
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
            Integer(50, 500, name='classifier__n_estimators'),
            Integer(3, 30, name='classifier__max_depth'),
            Integer(2, 20, name='classifier__min_samples_split'),
            Integer(1, 10, name='classifier__min_samples_leaf'),
            Categorical(['sqrt', 'log2', None], name='classifier__max_features'),
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
        
        # Plot cross-validation results
        if 'param_classifier__n_estimators' in cv_results.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(cv_results['param_classifier__n_estimators'], 
                    cv_results['mean_test_score'], 
                    marker='o', linestyle='-', label='Test Score')
            plt.plot(cv_results['param_classifier__n_estimators'], 
                    cv_results['mean_train_score'], 
                    marker='x', linestyle='--', label='Train Score')
            plt.xlabel('Number of Trees', fontsize=12)
            plt.ylabel('F2 Score', fontsize=12)
            plt.title('Cross-Validation Scores vs. Number of Trees', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    except:
        print("⚠️ Could not plot cross-validation results")
    
    # Get feature names after preprocessing for feature importance visualization
    ohe = preprocessor.named_transformers_['cat']
    cat_features = preprocessor.transformers_[1][2]  # categorical features
    try:
        if hasattr(ohe, 'get_feature_names_out'):
            cat_feature_names = ohe.get_feature_names_out(cat_features)
        else:
            cat_feature_names = [f"{feature}_{category}" for i, feature in enumerate(cat_features) 
                                for category in ohe.categories_[i]]
        
        num_features = preprocessor.transformers_[0][2]  # numerical features
        processed_feature_names = list(num_features) + list(cat_feature_names)
        
        # Extract the actual Random Forest model from the pipeline
        rf_model = best_model.named_steps['classifier']
        
        # Plot feature importance
        utils.plot_feature_importance(rf_model, processed_feature_names, "Random Forest")
    except:
        print("⚠️ Could not extract feature names for importance visualization")
    
    print(f"\n⏱️ Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the model
    model_filename = utils.save_model(best_model, "RandomForest", search.best_params_)
    
    return best_model