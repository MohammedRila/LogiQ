import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Autoencoder Definition
# -----------------------------
class AE(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# -----------------------------
# Training Function
# -----------------------------
def train_autoencoder(X, epochs=50, lr=1e-3, hidden=128, batch_size=32, patience=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    wait = 0

    for ep in range(epochs):
        epoch_loss = 0
        for batch in loader:
            x_batch = batch[0].to(device)
            opt.zero_grad()
            x_hat = model(x_batch)
            loss = loss_fn(x_hat, x_batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(X)
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler

# -----------------------------
# Reconstruction Error
# -----------------------------
def reconstruction_error(model, X, scaler, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        x_hat = model(X_t)
        err = ((X_t - x_hat) ** 2).mean(dim=1).cpu().numpy()
    return err

# -----------------------------
# Anomaly Detection
# -----------------------------
def detect_anomalies(model, X, scaler, threshold=None, device=None):
    err = reconstruction_error(model, X, scaler, device)
    if threshold is None:
        # adaptive threshold at 80th percentile
        threshold = np.percentile(err, 80)
    anomalies = err > threshold
    return anomalies, err, threshold

# -----------------------------
# Save & Load Model
# -----------------------------
def save_model(model, path: str, scaler=None):
    torch.save({'state_dict': model.state_dict(), 'scaler': scaler}, path)

def load_model(path: str, dim: int, hidden: int = 128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(dim, hidden).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    scaler = checkpoint.get('scaler', None)
    model.eval()
    return model, scaler

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Sample data (replace with TF-IDF or embeddings from logs)
    X = np.random.rand(100, 20)

    # Train
    model, scaler = train_autoencoder(X, epochs=50, batch_size=16)

    # Detect anomalies
    anomalies, errors, threshold = detect_anomalies(model, X, scaler)
    print(f"Threshold: {threshold}")
    print(f"Anomalies detected: {anomalies.sum()} / {len(X)}")
