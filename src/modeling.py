import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, confusion_matrix, 
    classification_report, make_scorer
)

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Splits data into features and target with stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify non-numeric columns and drop them (like IDs or strings not handled by encoder)
    # Note: Categorical columns should have been encoded in Task 1, 
    # but we'll ensure only numeric data reaches the model.
    X_numeric = X.select_dtypes(include=[np.number, 'bool'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_baseline(X_train, y_train, random_state=42):
    """Trains a Logistic Regression baseline model."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_ensemble(X_train, y_train, model_type='rf', **kwargs):
    """Trains an ensemble model (Random Forest or XGBoost)."""
    random_state = kwargs.get('random_state', 42)
    
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        raise ValueError("model_type must be 'rf' or 'xgboost'")
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns key metrics."""
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return {
        'auc_pr': auc_pr,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

def perform_cross_validation(model, X, y, cv=5):
    """Performs stratified k-fold cross-validation."""
    scoring = {
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    return {
        'f1_mean': np.mean(results['test_f1']),
        'f1_std': np.std(results['test_f1']),
        'precision_mean': np.mean(results['test_precision']),
        'recall_mean': np.mean(results['test_recall'])
    }
