import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data
df = pd.read_csv('data/solarwinds-ace-compiled/smooth_solarwinds_15min_metrics.csv', sep=";")  # Replace with your file path

print(df.columns)

# Drop 'Date' and use all other columns except 'Kp' as features
X = df.drop(columns=["Date", "Kp"]).values
Y = df["Kp"].values

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

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, Y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Save the model and scaler for later use
torch.save(model.state_dict(), 'kp_predictor.pth')
import joblib
joblib.dump(scaler, 'scaler.pkl')