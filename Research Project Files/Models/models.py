from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

###############################################################
# This file is exclusive to defining the models and their hyperparameters.
# It is imported into the model tuning script to create the models with the specified parameters.
###############################################################

def xg_boost():
    return XGBClassifier(
        learning_rate=0.05,
        max_depth=3,
        scale_pos_weight=15,
        eval_metric='logloss',
        random_state=42
    )

def random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )