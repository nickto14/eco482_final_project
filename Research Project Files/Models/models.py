###############################################################
# Read Me: 
#
# This file is for defining the models and their hyperparameters
#
###############################################################

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier


from xgboost import XGBClassifier

###############################################################
# Load in Data  
###############################################################

df = pd.read_csv("D:/Users/nickt/ECO482/Research Project Files/Data/FRED_MD_Cleaned_Transformed.csv")

y = df['Target_6M']
X = df.drop(columns=['Target_6M', 'sasdate', 'Inversion', 'Slope'])

###############################################################
# Random Walk
###############################################################

df['random_walk'] = df['Inversion']

###############################################################
# XGBoost and Random Forest 
###############################################################

tscv = TimeSeriesSplit(n_splits=3)
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
rf = RandomForestClassifier(random_state=42)

###############################################################
# Hyperparameter Optimization via GridSearchCV
###############################################################

# Define hyperparameter grids to search over
param_grids = {'xgboost':
    {"n_estimators": [100, 300],
    "max_depth": [3, 4, 6],           # XGBoost trees should generally be shallower than RF's
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9],          # row sampling per tree — helps prevent overfitting
    "colsample_bytree": [0.7, 0.9],   # feature sampling per tree — helps with your many correlated features
    "scale_pos_weight": [0.13]     # set to (# negative / # positive) for class imbalance (82 / 618),
    },
               
    'random_forest':     
    {"n_estimators": [200, 500],       # more trees = more stable, diminishing returns past ~500
    "max_depth": [4, 6, 8, None],     # shallower = less overfitting on a small dataset
    "min_samples_leaf": [5, 10, 20],  # higher = smoother, less likely to memorize noise
    "max_features": ["sqrt", 0.5],    # limits features considered per split
    "class_weight": ["balanced"]      # important given inversion class imbalance}
    }
}

models = {'xgboost': xgb,'random_forest': rf}

best_models = {}

for model_name, model in models.items():
    print(f"Starting Grid Search for {model_name}...")
    
    # Initialize GridSearchCV with the 3-fold TimeSeriesSplit
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grids[model_name],
        cv = tscv,
        scoring = 'recall',
        n_jobs = -1,
        verbose = 1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    # Record and display the results
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation Recall Score: {grid_search.best_score_:.3f}\n")


