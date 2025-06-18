"""XGBoost model with advanced optimization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import time
import utils
from config import *

def train_xgboost(X_train, y_train, preprocessor, feature_names):
    """
    Train an XGBoost model with advanced hyperparameter tuning
    """
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Create base pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=RANDOM_STATE, 
            use_label_encoder=False, 
            eval_metric='logloss',
            n_jobs=-1
        ))
    ])
    
    # Handle class imbalance with SMOTE if configured
    pipeline = utils.handle_imbalance(pipeline)
    
    # Set up parameter search based on optimization method
    if OPTIMIZATION_METHOD == 'grid':
        print("Using Grid Search for hyperparameter optimization...")
        
        # Define hyperparameters to tune
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            'classifier__gamma': [0, 0.1, 0.2],
            'classifier__scale_pos_weight': [1, 3, 5]  # Helps with class imbalance
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
            'classifier__learning_rate': np.logspace(-3, 0, 100),
            'classifier__max_depth': np.arange(3, 15, 1),
            'classifier__min_child_weight': np.arange(1, 10, 1),
            'classifier__subsample': np.linspace(0.6, 1.0, 10),
            'classifier__colsample_bytree': np.linspace(0.6, 1.0, 10),
            'classifier__gamma': np.linspace(0, 0.5, 10),
            'classifier__scale_pos_weight': np.linspace(1, 10, 10)
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
            Real(0.001, 0.3, prior='log-uniform', name='classifier__learning_rate'),
            Integer(3, 12, name='classifier__max_depth'),
            Integer(1, 10, name='classifier__min_child_weight'),
            Real(0.6, 1.0, name='classifier__subsample'),
            Real(0.6, 1.0, name='classifier__colsample_bytree'),
            Real(0.0, 0.5, name='classifier__gamma'),
            Real(1.0, 10.0, name='classifier__scale_pos_weight')
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
        
        # Plot learning rate vs score if available
        if 'param_classifier__learning_rate' in cv_results.columns:
            plt.figure(figsize=(10, 6))
            plt.semilogx(cv_results['param_classifier__learning_rate'], 
                        cv_results['mean_test_score'], 
                        marker='o', linestyle='-', label='Test Score')
            plt.semilogx(cv_results['param_classifier__learning_rate'], 
                        cv_results['mean_train_score'], 
                        marker='x', linestyle='--', label='Train Score')
            plt.xlabel('Learning Rate (log scale)', fontsize=12)
            plt.ylabel('F2 Score', fontsize=12)
            plt.title('Cross-Validation Scores vs. Learning Rate', fontsize=14)
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
        
        # Extract the actual XGBoost model from the pipeline
        xgb_model = best_model.named_steps['classifier']
        
        # Plot feature importance
        utils.plot_feature_importance(xgb_model, processed_feature_names, "XGBoost")
    except:
        print("⚠️ Could not extract feature names for importance visualization")
    
    print(f"\n⏱️ Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the model
    model_filename = utils.save_model(best_model, "XGBoost", search.best_params_)
    
    return best_model