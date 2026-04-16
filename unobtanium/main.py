# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# --- Load Data ---
def load_real_data():
    # Load Source 1: Date, X
    source1 = pd.read_csv('data/mag-kiruna-compiled/smooth_normalized.csv', delimiter=";", nrows=100000)
    source1['Date'] = pd.to_datetime(source1['Date'], format="%Y%m%d%H%M%S")
    source1 = source1.groupby(pd.Grouper(key='Date', freq='15min')).mean(numeric_only=True).reset_index()

    # Load Source 2: Date, Speed, Density, Bt, Bz
    source2 = pd.read_csv('data/solarwinds-ace-compiled/smooth_normalized.csv', delimiter=";", nrows=100000)
    source2['Date'] = pd.to_datetime(source2['Date'], format="%Y%m%d%H%M%S")
    source2 = source2.groupby(pd.Grouper(key='Date', freq='15min')).mean(numeric_only=True).reset_index()

    # Load Correction Source: Date, Correction
    correction = pd.read_csv('data/kp-compiled/smooth.csv', delimiter=";", nrows=100000)
    correction['Date'] = pd.to_datetime(correction['Date'], format="%Y%m%d%H%M%S")
    correction = correction.groupby(pd.Grouper(key='Date', freq='15min')).mean(numeric_only=True).reset_index()

    # Merge all sources on Date
    data = pd.merge(source1, source2, on='Date', how='inner')
    data = pd.merge(data, correction, on='Date', how='inner')

    # Fill NaN values with the mean of each column
    data = data.fillna(data.mean())

    # Convert to list of tuples for the dataset
    data_list = data.to_dict('records')
    data = [
        (
            row['Date'],
            float(row['X']),
            float(row['Speed']),
            float(row['Density']),
            float(row['Bt']),
            float(row['Bz']),
            float(row['Kp'])
        )
        for row in data_list
    ]

    if data:
        print("Sample data entry:", data[0])
        print("Number of entries:", len(data))
    else:
        print("No data found! Check your CSV files and paths.")

    return data

data = load_real_data()

# --- Dataset ---
class SolarWindDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        date, X, Speed, Density, Bt, Bz, correction = self.data[idx]
        features = np.array([X, Speed, Density, Bt, Bz], dtype=np.float32)
        correction = int(correction)
        return features, correction  # Return class index, not one-hot

# --- Collate Function ---
def collate_fn(batch):
    features, corrections = zip(*batch)
    features = [torch.tensor(f) for f in features]
    features = torch.stack(features).unsqueeze(1)  # Shape: (batch_size, 1, input_size)
    corrections = torch.tensor(corrections, dtype=torch.long)  # Class indices, not one-hot
    return features, corrections

# --- Model ---
class SolarWindLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
    
def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_features, _ in dataloader:
            outputs = model(batch_features)
            predictions.append(outputs.numpy())
    return np.concatenate(predictions, axis=0)

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy


# --- Main ---
if __name__ == "__main__":
    dataset = SolarWindDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    model = SolarWindLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0017)

    for epoch in range(4000):
        model.train()
        total_loss = 0
        total_acc = 0
        for batch_features, batch_targets in dataloader:
            # Check for NaN in input
            if torch.isnan(batch_features).any():
                print("NaN in input!")
                continue

            optimizer.zero_grad()
            outputs = model(batch_features)

            # Check for NaN in output
            if torch.isnan(outputs).any():
                print("NaN in output!")
                continue

            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, batch_targets)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {total_acc/len(dataloader):.4f}')

        if (epoch + 1) % 500 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': total_loss,
            }, f"model/checkpoint_epoch_{epoch + 1}.pt")