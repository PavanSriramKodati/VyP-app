import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# CONFIG
FILE_NAME = "..\client_0.csv" # Check your training file
LABEL_COLUMN = "label"
# Use the EXACT same drop list as your task.py
DROP_COLS = ['src_ip', 'dst_ip', 'timestamp', 'flow_id', 'simillarhttp', 
             'dst_port','src_port','protocol','subflow_fwd_pkts',
             'flow_duration','flow_iat_mean','flow_iat_max','flow_iat_std','flow_byts_s', 'flow_pkts_s',
             'fwd_iat_tot','fwd_iat_max','fwd_iat_mean','fwd_header_len','fwd_iat_std','fwd_pkts_s',
             'bwd_iat_tot','bwd_iat_max','bwd_iat_mean','bwd_iat_std',
             'active_max','active_min','active_mean','active_std',
             'idle_max','idle_min','idle_mean','idle_std',
             'tot_fwd_pkts','init_fwd_win_byts','totlen_fwd_pkts','subflow_fwd_byts','pkt_size_avg','fwd_pkt_len_mean','pkt_len_mean','fwd_pkt_len_max','fwd_seg_size_avg','pkt_len_max','pkt_len_min'
             ]
def find_the_leak():
    print(f"Loading {FILE_NAME}...")
    df = pd.read_csv(FILE_NAME)
    
    # 1. Clean EXACTLY like task.py
    y = df[LABEL_COLUMN]
    X = df.drop(columns=[LABEL_COLUMN])
    
    # Drop known cols
    for col in DROP_COLS:
        matching = [c for c in X.columns if c.lower() == col.lower()]
        if matching:
            X = X.drop(columns=matching)

    # Encode strings
    print("Encoding text columns...")
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f" -> Encoding {col}")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 2. Train a Decision Tree (It finds the easiest path to 100%)
    print("Training Diagnostic Tree...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # 3. Get Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "="*40)
    print("TOP FEATURES USED BY THE MODEL")
    print("="*40)
    
    # Check if the top feature is suspicious
    top_feature = X.columns[indices[0]]
    top_score = importances[indices[0]]
    
    for f in range(5):
        if f < len(X.columns):
            name = X.columns[indices[f]]
            score = importances[indices[f]]
            print(f"{f+1}. {name}: {score:.4f} ({score*100:.2f}%)")

    print("\n" + "="*40)
    if top_score > 0.90:
        print(f"⚠️  WARNING: '{top_feature}' is a LEAK!")
        print(f"The model is using '{top_feature}' to get 100% accuracy.")
        print("Add this column to your DROP_COLS list in task.py.")
    else:
        print("No single feature is dominating. The model is using a mix.")
        print("The 1.0 accuracy might be genuine (or the test set is identical to train).")

if __name__ == "__main__":
    find_the_leak()