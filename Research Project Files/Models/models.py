###############################################################
# Read Me: 
#
# This file is for defining the models and their hyperparameters
#
###############################################################

import pickle

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
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

tscv = TimeSeriesSplit(n_splits=5)
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
rf = RandomForestClassifier(random_state=42)

###############################################################
# Hyperparameter Optimization via GridSearchCV
###############################################################

# Define hyperparameter grids to search over
param_grids = {'XGBoost':
    {"n_estimators": [100, 300],
    "max_depth": [3, 4, 6],           # XGBoost trees should generally be shallower than RF's
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9],          # row sampling per tree — helps prevent overfitting
    "colsample_bytree": [0.7, 0.9],   # feature sampling per tree — helps with your many correlated features
    "scale_pos_weight": [0.13]     # set to (# negative / # positive) for class imbalance (82 / 618),
    },
               
    'Random Forest':     
    {"n_estimators": [200, 500],       # more trees = more stable, diminishing returns past ~500
    "max_depth": [4, 6, 8, None],     # shallower = less overfitting on a small dataset
    "min_samples_leaf": [5, 10, 20],  # higher = smoother, less likely to memorize noise
    "max_features": ["sqrt", 0.5],    # limits features considered per split
    "class_weight": ["balanced"]      # important given inversion class imbalance}
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
        scoring = 'recall',
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
        'best_cv_recall': grid_search.best_score_
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

        y_pred = (y_proba >= 0.5).astype(int)

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
    print(f"Best cross-validation Recall Score: {grid_search.best_score_:.3f}\n")

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

