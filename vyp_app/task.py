import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------- CONFIGURATION ----------------------------
INPUT_SIZE = 79  
LABEL_COLUMN = "label" 
DROP_COLS = ['src_ip', 'dst_ip', 'timestamp']
TEST_FILE = "test.csv"

# ----------------------------  MULTILAYER PERCEPTRON ----------------------------
# Set up MLP with 2 hidden layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Layer 1: 64 neurons, reLU activation
        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        # Layer 2: 32 neurons, reLU activation
        self.fc2 = nn.Linear(64, 32)
        # Output layer
        # 3 outputs: DoS, MITM, Normal
        self.fc3 = nn.Linear(32, 3) 

    # Apply 3 layers to return raw scores
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------  DATA PREPARATION ----------------------------
def get_clean_features_and_labels(df):
    # Drop NaN rows
    df = df.dropna(subset=[LABEL_COLUMN])

    # Convert label column into NumPy array of strings
    y = df[LABEL_COLUMN].astype(str).values
  
    # Drop label from feature set
    df_features = df.drop(columns=[LABEL_COLUMN])

    # Drop columns from the DataFrame
    for col in DROP_COLS:
        matching_cols = [c for c in df_features.columns if c.lower() == col.lower()]
        if matching_cols:
            df_features = df_features.drop(columns=matching_cols)

    # Encode categorical columns
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
    
    print("-" * 50)
    print("First 10 rows of df_features:")
    print(df_features.head())
            
    return df_features, y

# ----------------------------  LOAD TRAINING & TESTING DATA ----------------------------
def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """
    For each client (partition_id), load its CSV (client_{id}.csv),
    then do an 80/20 train/validation split on that client's data.
    """
    train_file = f"client_{partition_id}.csv"
    print(f"Client {partition_id}: Loading data from {train_file}...")

    try:
        df_train = pd.read_csv(train_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e}")

    # Clean data: drop label & unwanted cols, encode categoricals
    X_df, y_raw = get_clean_features_and_labels(df_train)

    print(f"\n{'-' * 50}")
    print(f"Shape of X_df: {X_df.shape}")
    print(f"Shape of y_raw: {y_raw.shape}")

    # Convert DataFrame to NumPy
    X = X_df.values

    # 80/20 split of this client's data
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X,
        y_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_raw,  # keep class distribution similar
    )

    print(f"\n{'-' * 50}")
    print(f"CLIENT {partition_id} FEATURE COUNT: {X_train.shape[1]}")
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    # Encode labels to integers (e.g., DoS / Normal / MITM -> 0 / 1 / 2)
    le_y = LabelEncoder()
    all_labels = np.unique(y_raw)
    le_y.fit(all_labels)

    print(
        f"\n{'-' * 50}\n"
        f"Encoded labels (client {partition_id}): "
        f"{dict(zip(le_y.transform(le_y.classes_), le_y.classes_))}"
    )

    y_train = le_y.transform(y_train_raw)
    y_val = le_y.transform(y_val_raw)

    # Scale features: fit on train, transform both train & val
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert NumPy arrays to PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    # DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

"""
def load_data(partition_id: int, num_partitions: int):
    
    train_file = f"client_{partition_id}.csv" 
    print(f"Client {partition_id}: Loading TRAIN data from {train_file}...")
    print(f"Client {partition_id}: Loading TEST data from {TEST_FILE}...")
    
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e}")

    # Clean data
    X_train_df, y_train_raw = get_clean_features_and_labels(df_train)
    X_test_df, y_test_raw = get_clean_features_and_labels(df_test)
    
    print("-" * 50)
    print(f"Shape of X_train_df: {X_train_df.shape}")
    print(f"Shape of y_train_raw: {y_train_raw.shape}")   

    # Convert each DataFrame into NumPy array
    X_train_df, X_test_df = X_train_df.align(X_test_df, join='inner', axis=1)
    X_train = X_train_df.values
    X_test = X_test_df.values

    print("-" * 50)
    print(f"CLIENT {partition_id} FEATURE COUNT: {X_train.shape[1]}")
    print("-" * 50)

    # Encode label to encode DoS, Normal, NMITM to 0, 1, 2
    le_y = LabelEncoder()

    # Fit on all unique labels
    all_labels = np.unique(np.concatenate((y_train_raw, y_test_raw)))
    le_y.fit(all_labels)
    
    print("-" * 50)
    print(f"Encoded labels: {dict(zip(le_y.transform(le_y.classes_), le_y.classes_))}")
    print("-" * 50)
    
    y_train = le_y.transform(y_train_raw)
    y_test = le_y.transform(y_test_raw)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert NumPy arrays to to PyTorch dataset objects
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.long)
    )

    # Feed PyTorch dataset objects to DataLoader
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, valloader
"""

# ----------------------------  GLOBAL TEST DATA (test.csv) ---------------------------- #
def load_global_testloader(batch_size: int = 32):
    print(f"\n{'-' * 50}")
    print(f"Loading GLOBAL TEST data from {TEST_FILE}...")

    try:
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing test file: {e}")

    # Clean data
    X_df, y_raw = get_clean_features_and_labels(df_test)

    print(f"\n{'-' * 50}")
    print(f"Shape of X_test_df: {X_df.shape}")
    print(f"Shape of y_test_raw: {y_raw.shape}")

    X = X_df.values

    # Encode labels
    le_y = LabelEncoder()
    all_labels = np.unique(y_raw)
    le_y.fit(all_labels)

    print(
        f"\n{'-' * 50}\n"
        f"Encoded labels (GLOBAL TEST): "
        f"{dict(zip(le_y.transform(le_y.classes_), le_y.classes_))}"
    )

    y = le_y.transform(y_raw)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return testloader

# ----------------------------  TRAINING MODEL ----------------------------
def train(net, trainloader, epochs, lr, device):
    # use GPU
    net.to(device)

    # Use CrossEntropy to calculate loss (for multiclass)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Use Adam optimizer to update model weights
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Train model
    net.train()
    running_loss = 0.0

    # Train in batches for each epoch
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            # MLP has 4 steps:
            # 1. Forward pass
            outputs = net(features)

            # 2. Calculate errror
            loss = criterion(outputs, labels)

            # 3. Use backpropagation
            loss.backward()

            # 4. Update right
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

# ----------------------------  TESTING MODEL ----------------------------
def test(net, testloader, device):
    # Use GPU
    net.to(device)

    # Use CrossEntropy to calculate loss (for multiclass)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    y_true = []
    y_pred = []
    
    # Test in batches for each epoch
    with torch.no_grad():
        for batch in testloader:
            # Convert features and labels to same GPU device
            features, labels = batch[0].to(device), batch[1].to(device)

            # 1. Forward pass
            outputs = net(features)

            # 2. Calculate loss
            loss += criterion(outputs, labels).item()

            # 3. Count the number of correct predictions 
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            # 4. Save correct predictions to Python lists
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # Calculate 1. accuracy & 2. loss
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    
    # # ---------------print(f"\n{'-' * 50}")-------------  PRINT RESULT ----------------------------
    print(f"\n{'-' * 50}")
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_true, y_pred))

    print(f"\n{'-' * 50}")
    print("REPORT:")
    print(classification_report(y_true, y_pred))

    print("-" * 50)
    return loss, accuracy