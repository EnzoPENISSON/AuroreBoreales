import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data
df = pd.read_csv('data/solarwinds-ace-compiled/smooth_solarwinds_15min_metrics.csv', sep=";")  # Replace with your file path

print(df.columns)

# Ignore Date entirely if present in the source file.
if "Date" in df.columns:
    df = df.drop(columns=["Date"])

# Remove rows with missing targets and any non-finite values before training.
feature_columns = [column for column in df.columns if column != "Kp"]
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_columns + ["Kp"])

if df.empty:
    raise ValueError("No valid rows left after removing missing or non-finite values.")

print(f"Rows kept for training: {len(df)}")

# Use all columns except Kp as features
X = df[feature_columns].to_numpy(dtype=np.float32)
Y = df["Kp"].to_numpy(dtype=np.float32)

if not np.isfinite(X).all() or not np.isfinite(Y).all():
    raise ValueError("Training data still contains non-finite values after cleaning.")

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.FloatTensor(Y_test).view(-1, 1)

# Define a simple neural network
class KpPredictor(nn.Module):
    def __init__(self, input_dim):
        super(KpPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
input_dim = X_train.shape[1]
model = KpPredictor(input_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_kp_accuracy(y_true_tensor, y_pred_tensor):
    y_true_classes = np.rint(y_true_tensor.numpy().flatten()).astype(int)
    y_pred_classes = np.rint(y_pred_tensor.numpy().flatten()).astype(int)
    return accuracy_score(y_true_classes, y_pred_classes)

# Training loop
epochs = 2500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train_tensor)
            test_predictions = model(X_test_tensor)
            train_accuracy = compute_kp_accuracy(Y_train_tensor, train_predictions)
            test_accuracy = compute_kp_accuracy(Y_test_tensor, test_predictions)

        print(
            f'Epoch [{epoch+1}/{epochs}], '
            f'Loss: {loss.item():.4f}, '
            f'Train Accuracy: {train_accuracy * 100:.2f}%, '
            f'Test Accuracy: {test_accuracy * 100:.2f}%'
        )

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, Y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

    accuracy = compute_kp_accuracy(Y_test_tensor, predictions)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model and scaler for later use
torch.save(model.state_dict(), 'kp_predictor.pth')
import joblib
joblib.dump(scaler, 'scaler.pkl')