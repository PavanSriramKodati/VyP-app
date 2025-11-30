"""VyP-app: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------- CONFIGURATION ---------------- #
# 1. Update this number after running once if needed
INPUT_SIZE = 77  
# 2. Your Label Name
LABEL_COLUMN = "label" 
# 3. Columns to drop
DROP_COLS = ['src_ip', 'dst_ip', 'timestamp', 'flow_id', 'simillarhttp', 'flow_byts_s', 'flow_pkts_s']
# 4. Test File Name
TEST_FILE = "test.csv"
# ----------------------------------------------- #

class Net(nn.Module):
    """Simple MLP for Tabular Data."""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def get_clean_features_and_labels(df):
    """Helper function to clean a specific dataframe."""
    
    # 1. Separate Y
    y = df[LABEL_COLUMN].values
    
    # 2. Drop Label from X
    df_features = df.drop(columns=[LABEL_COLUMN])

    # 3. Drop Identifiers
    for col in DROP_COLS:
        matching_cols = [c for c in df_features.columns if c.lower() == col.lower()]
        if matching_cols:
            df_features = df_features.drop(columns=matching_cols)

    # 4. Encode Categorical Columns (Protocol, etc.)
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
            
    return df_features, y

def load_data(partition_id: int, num_partitions: int):
    """Load Training Data (Client) and Test Data (Global)."""
    
    # --- 1. LOAD DATA ---
    train_file = f"client_{partition_id}.csv" 
    print(f"Client {partition_id}: Loading TRAIN data from {train_file}...")
    print(f"Client {partition_id}: Loading TEST data from {TEST_FILE}...")
    
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e}")

    # --- 2. CLEAN BOTH DATASETS ---
    # We clean them separately but using the same logic
    X_train_df, y_train_raw = get_clean_features_and_labels(df_train)
    X_test_df, y_test_raw = get_clean_features_and_labels(df_test)

    # --- 3. ALIGN COLUMNS ---
    # Critical: Ensure Train and Test have exact same columns in same order
    # If Test has extra columns, drop them. If missing, fill 0.
    X_train_df, X_test_df = X_train_df.align(X_test_df, join='inner', axis=1)
    
    X_train = X_train_df.values
    X_test = X_test_df.values

    # !!! DEBUG PRINT !!!
    print(f"----------------------------------------------------")
    print(f"CLIENT {partition_id} FINAL FEATURE COUNT: {X_train.shape[1]}")
    print(f"Please ensure INPUT_SIZE in task.py is set to: {X_train.shape[1]}")
    print(f"----------------------------------------------------")

    # --- 4. ENCODE LABELS ---
    # We fit the encoder on combined labels to ensure 0 and 1 mean the same thing in both
    le_y = LabelEncoder()
    # Fit on all possible labels to avoid errors if one file misses a class
    all_labels = np.unique(np.concatenate((y_train_raw, y_test_raw)))
    le_y.fit(all_labels)
    
    y_train = le_y.transform(y_train_raw)
    y_test = le_y.transform(y_test_raw)

    # --- 5. SCALE ---
    # IMPORTANT: Fit scaler ONLY on Training data, then transform Test data
    # This prevents "looking into the future" (Data Leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 6. TO TENSORS ---
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.long)
    )

    # Return Loaders
    # Note: We use the external test set as 'valloader' for the simulation
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, valloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy