import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CONFIG
FILE_NAME = "client_0.csv"  # Use one of your client files
LABEL_COLUMN = "label"
DROP_COLS = ['src_ip', 'dst_ip', 'timestamp', 'flow_id', 'simillarhttp', 'flow_byts_s', 'flow_pkts_s']

def check_feature_importance():
    # 1. Load Data
    print(f"Loading {FILE_NAME}...")
    df = pd.read_csv(FILE_NAME)
    
    # 2. Clean (Same logic as your task.py)
    y = df[LABEL_COLUMN]
    X = df.drop(columns=[LABEL_COLUMN])
    
    # Drop the columns you dropped in task.py
    for col in DROP_COLS:
        cols_to_drop = [c for c in X.columns if c.lower() == col.lower()]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            
    # Encode strings
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 3. Train a quick Random Forest
    print("Training diagnostic model...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    # 4. Get Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- TOP 5 FEATURES RESPONSIBLE FOR 100% ACCURACY ---")
    for f in range(5):
        feature_name = X.columns[indices[f]]
        importance_score = importances[indices[f]]
        print(f"{f+1}. {feature_name}: {importance_score:.4f} ({importance_score*100:.1f}%)")

    print("\n----------------------------------------------------")
    if importances[indices[0]] > 0.40:
        print("DIAGNOSIS: One feature is dominating.")
        print(f"The model is likely just checking '{X.columns[indices[0]]}' to decide.")
    else:
        print("DIAGNOSIS: The model is using a mix of features.")

if __name__ == "__main__":
    check_feature_importance()