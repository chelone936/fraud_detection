from imblearn.over_sampling import SMOTE
import pandas as pd

def handle_imbalance(df, target_col):
    """Applies SMOTE to balance the target class."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify non-numeric columns for SMOTE (it requires numeric input)
    # In a real pipeline, these should be encoded first.
    X_numeric = X.select_dtypes(include=['number'])
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_numeric, y)
    
    balanced_df = pd.concat([X_res, y_res], axis=1)
    return balanced_df
