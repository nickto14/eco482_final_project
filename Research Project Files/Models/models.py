import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix,
                             precision_recall_fscore_support, roc_curve, auc,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

###############################################################
# This script performs the full modeling pipeline for the XGBoost Classifier, including:
# - Data loading and preparation
# - TimeSeriesSplit cross-validation
# - Model training and evaluation
###############################################################

###############################################################
# Read in the cleaned dataset and prepare the 
# feature matrix X and target vector y
###############################################################

df = pd.read_csv("D:/Users/nickt/ECO482/Research Project Files/Cleaning/FRED_MD_Cleaned.csv")
X = df.drop(columns=['sasdate', 'Slope', 'Inversion', 'Target_6M'])
y = df['Target_6M']

###############################################################
# Set up TimeSeriesSplit for cross-validation 
###############################################################

tscv = TimeSeriesSplit(n_splits = 3)

all_importances = []
results_list = []
fold = 1

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scaling Data
    scaler = preprocessing.StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    # Stage 1: XGBoost (Tuned for 123-Feature Space)
    gbc = XGBClassifier(
        learning_rate=0.05,
        max_depth=3,
        reg_alpha=0,
        reg_lambda=1.0,
        scale_pos_weight=15,
        eval_metric='logloss',
        random_state=42
    )

    # Train once, use for both metrics and importance
    gbc.fit(x_train_scaled, y_train)

    # Stage 2: Predictions
    y_proba = gbc.predict_proba(x_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.10).astype(int)

    # Handle cases with no inversions in the test window (Fold 2)
    if len(np.unique(y_test)) > 1:
        current_pr_auc = average_precision_score(y_test, y_proba)
        current_roc_auc = roc_auc_score(y_test, y_proba)
    else:
        current_pr_auc, current_roc_auc = np.nan, np.nan

    # Metrics for Class 1.0 (Inversions)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=[1], zero_division=0)

    # Store results
    results_list.append({
        'Fold': fold,
        'Features': x_train_scaled.shape[1],
        'Accuracy': accuracy_score(y_test, y_pred),
        'PR-AUC': current_pr_auc,
        'ROC-AUC': current_roc_auc,
        'Recall_Class1': r[0],
        'Precision_Class1': p[0],
    })

    # Store feature importance for overall aggregation
    all_importances.append(gbc.feature_importances_)

    print(f"Fold {fold} Complete. ROC-AUC: {current_roc_auc:.4f}")
    fold += 1


