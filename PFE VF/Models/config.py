"""Configuration settings for the churn prediction models"""

# Data settings
DATA_PATH = "Book1.xlsx"
TARGET_COLUMN = 'Churn'  # Modify based on your data
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cross-validation settings
CV_FOLDS = 5
CV_STRATEGY = 'stratified'  # Options: 'stratified', 'time_series', 'nested'

# Preprocessing settings
HANDLE_IMBALANCE = True
IMBALANCE_METHOD = 'smote'  # Options: 'smote', 'adasyn', 'class_weight'
SCALING_METHOD = 'standard'  # Options: 'standard', 'robust', 'minmax'

# Optimization settings
N_ITER = 50  # Number of iterations for Bayesian optimization
OPTIMIZATION_METHOD = 'bayesian'  # Options: 'grid', 'random', 'bayesian'
SCORING = 'f2'  # Primary metric to optimize

# Feature engineering
FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'rfe'  # Options: 'rfe', 'boruta', 'shap'
TOP_N_FEATURES = 15

# Ensemble settings
USE_ENSEMBLE = True
ENSEMBLE_METHOD = 'stacking'  # Options: 'voting', 'stacking', 'blending'