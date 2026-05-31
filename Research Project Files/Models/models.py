from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

###############################################################
# This file is exclusive to defining the models and their hyperparameters.
# It is imported into the model tuning script to create the models with the specified parameters.
###############################################################

def xg_boost():
    return XGBClassifier(
        learning_rate=2,
        max_depth=3,
        reg_alpha=1.5,
        reg_lambda=5.0,
        scale_pos_weight=15,
        eval_metric='logloss',
        random_state=42,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        device='cuda'
    )


def random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf = 4, 
        min_samples_split = 10,
        class_weight='balanced',
        random_state=42
    )