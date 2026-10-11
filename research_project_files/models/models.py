###############################################################
# Read Me: 
#
# This file is for defining the models and their hyperparameters
#
###############################################################

import pickle

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

###############################################################
# Load in Data  
###############################################################

df = pd.read_csv("D:/Users/nickt/ECO482/Research Project Files/Data/FRED_MD_Cleaned_Transformed.csv")

###############################################################
# Define variables 
###############################################################

y = df['Target_6M']
X = df.drop(columns=['Target_6M', 'sasdate', 'Slope'])

###############################################################
# XGBoost and Random Forest 
###############################################################

tscv = TimeSeriesSplit(n_splits=2)
xgb = XGBClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

###############################################################
# Hyperparameter Optimization via GridSearchCV
###############################################################

param_grids = {
    'XGBoost': {
        "max_depth": [2, 3, 5],                    
        "learning_rate": [0.05, 0.1, 0.3, 0.5],
        "colsample_bytree": [0.7, 0.9, 1],          
        "reg_alpha": [2, 5, 10, 20],
        "reg_lambda": [2, 5, 10, 20],
        "scale_pos_weight": [6.5, 10],
        "max_delta_step": [1]
    },

    'Random Forest': {
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [3, 5, 10],
        "max_features": ["sqrt", 0.5],
        "class_weight": ['balanced', 'balanced_subsample']
    }
}

models = {'XGBoost': xgb,'Random Forest': rf}

best_models = {}
results_summary = []
evaluation_results = []

for model_name, model in models.items():
    print(f"Starting Grid Search for {model_name}...")
    
    # Initialize GridSearchCV with the 3-fold TimeSeriesSplit
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grids[model_name],
        cv = tscv,
        scoring = 'f1',
        n_jobs = -1,
        verbose = 1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    # Record and display the results
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    results_summary.append({
        'model': model_name,
        'best_params': grid_search.best_params_,
        'best_cv_avg_precision': grid_search.best_score_
    })

    # Evaluate the best model on each time-series fold and save the outputs
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_fold = clone(best_model)
        model_fold.fit(X_train, y_train)

        if hasattr(model_fold, 'predict_proba'):
            y_proba = model_fold.predict_proba(X_test)[:, 1]
        else:
            y_proba = model_fold.decision_function(X_test)

        y_pred = (y_proba >= 0.45).astype(int)

        evaluation_results.append({
            'model': model_name,
            'fold': fold,
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba),
            'y_true': y_test.to_numpy(),
            'y_pred': y_pred,
            'y_proba': y_proba,
        })

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation Average Precision Score: {grid_search.best_score_:.3f}\n")

###############################################################
# Random Walk Baseline (evaluated on same folds as XGB/RF)
###############################################################

rw_evaluation_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    y_test = y.iloc[test_idx]
    rw_pred = (X['Inversion'].iloc[test_idx] > 0).astype(int)
    rw_evaluation_results.append({
        'model': 'Random Walk',
        'fold': fold,
        'accuracy': accuracy_score(y_test, rw_pred),
        'recall': recall_score(y_test, rw_pred, zero_division=0),
        'precision': precision_score(y_test, rw_pred, zero_division=0),
        'y_true': y_test.to_numpy(),
        'y_pred': rw_pred.to_numpy(),
    })
evaluation_results.extend(rw_evaluation_results)

###############################################################
# Export the models and results to a pickle file
###############################################################

summary_df = pd.DataFrame(results_summary)
output_dir = "D:/Users/nickt/ECO482/Research Project Files/Models"
output_path = f"{output_dir}/trained_models.pkl"

with open(output_path, 'wb') as f:
    pickle.dump({
        'best_models': best_models,
        'summary_df': summary_df,
        'evaluation_results': evaluation_results,
        'X': X,
        'y': y,
        'feature_names': X.columns.tolist()
    }, f)

