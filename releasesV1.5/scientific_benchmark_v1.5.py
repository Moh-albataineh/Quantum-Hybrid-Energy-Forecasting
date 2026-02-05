import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
import time
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from data_processor_v2 import load_and_process_data

# ==========================================
# ⚙️ GLOBAL CONFIGURATION (V1.5.1 FIXED)
# ==========================================
SEEDS = [42, 101, 777, 2026, 99]
SEQ_LENGTH = 168
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folds: Spring/Summer, Fall, Winter
FOLDS = [
    {'train_end': 100000, 'val_size': 168*2, 'test_size': 168*2}, 
    {'train_end': 120000, 'val_size': 168*2, 'test_size': 168*2},
    {'train_end': 140000, 'val_size': 168*2, 'test_size': 168*2}
]

LOG_FILE = "v1.5_results_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ==========================================
# 🧠 MODELS
# ==========================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(hidden_size, n_qubits)
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (2, n_qubits, 3)})
        self.fc = nn.Linear(n_qubits, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.proj(out[:, -1, :])
        out = torch.tanh(out) * (np.pi / 2)
        out = self.q_layer(out)
        return self.fc(out)

# ==========================================
# 🛠️ UTILITIES
# ==========================================

def get_data_split(features, fold_cfg):
    train_end = fold_cfg['train_end']
    val_end = train_end + fold_cfg['val_size']
    test_end = val_end + fold_cfg['test_size']
    
    train_raw = features[:train_end]
    val_raw = features[train_end:val_end]
    test_raw = features[val_end:test_end]
    
    scaler_load = MinMaxScaler((0, 1))
    scaler_load.fit(train_raw[:, 0].reshape(-1, 1))
    
    def scale_and_seq(data, scaler):
        d_copy = data.copy()
        d_copy[:, 0] = scaler.transform(d_copy[:, 0].reshape(-1, 1)).flatten()
        Xs, ys = [], []
        for i in range(len(d_copy) - SEQ_LENGTH):
            Xs.append(d_copy[i:i+SEQ_LENGTH])
            ys.append(d_copy[i+SEQ_LENGTH, 0])
        # Reshape Y to (N, 1) immediately to fix broadcasting bug
        return np.array(Xs), np.array(ys).reshape(-1, 1)

    X_train, y_train = scale_and_seq(train_raw, scaler_load)
    X_val, y_val = scale_and_seq(val_raw, scaler_load)
    X_test, y_test = scale_and_seq(test_raw, scaler_load)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_load, test_raw[:, 0]

def train_one_model(model, train_loader, val_loader, epochs, lr, patience=5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            # FIX: Ensure target shape matches prediction shape
            loss = criterion(pred, yb) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_weights)
    return model

# ==========================================
# 🔎 HYPERPARAMETER TUNING
# ==========================================
def run_tuning(features):
    log("\n🔎 STARTING HYPERPARAMETER TUNING (Fold 1 Only)...")
    (X_tr, y_tr), (X_val, y_val), _, _, _ = get_data_split(features, FOLDS[0])
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr[-5000:], dtype=torch.float32), 
                                          torch.tensor(y_tr[-5000:], dtype=torch.float32)), 
                            batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                        torch.tensor(y_val, dtype=torch.float32)), 
                          batch_size=64)
    
    grid = [
        {'hidden': 64, 'lr': 0.001},
        {'hidden': 64, 'lr': 0.0005},
        {'hidden': 128, 'lr': 0.001},
        {'hidden': 128, 'lr': 0.0005}
    ]
    
    best_cfg = None
    best_score = float('inf')
    
    input_dim = X_tr.shape[2]
    
    for cfg in grid:
        log(f"   Testing: {cfg} ...")
        torch.manual_seed(42) 
        model = LSTMModel(input_dim, cfg['hidden']).to(DEVICE)
        trained_model = train_one_model(model, train_loader, val_loader, 10, cfg['lr'], patience=3)
        
        loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = trained_model(xb)
                loss += nn.MSELoss()(pred, yb).item()
        
        if loss < best_score:
            best_score = loss
            best_cfg = cfg
            
    log(f"✅ Best Config Found: {best_cfg}")
    return best_cfg

# ==========================================
# 🧪 MAIN EXPERIMENT LOOP
# ==========================================
def run_v1_5():
    log("🚀 STARTING V1.5.1: FIXED SHAPES & PROTOCOL")
    log("="*60)
    
    df = pd.read_csv('PJME_hourly.csv', index_col='Datetime', parse_dates=True)
    df.sort_index(inplace=True)
    
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    features_raw = df[['PJME_MW', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values.astype(float)
    input_dim = 5
    
    # 1. Tune
    best_cfg = run_tuning(features_raw)
    HIDDEN = best_cfg['hidden']
    LR = best_cfg['lr']
    
    results = {'Classical': [], 'Hybrid': [], 'Naive': []}
    
    # 2. Rolling Loop
    for fold_idx, fold_cfg in enumerate(FOLDS):
        log(f"\n📅 FOLD {fold_idx+1}/{len(FOLDS)} (Train End: {fold_cfg['train_end']})")
        log("-" * 40)
        
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), scaler, y_test_true_raw = get_data_split(features_raw, fold_cfg)
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), 
                                  batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), 
                                batch_size=1024)
        test_tensor = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        
        # Naive Baseline
        naive_pred_scaled = X_te[:, -1, 0].reshape(-1, 1)
        naive_pred = scaler.inverse_transform(naive_pred_scaled).flatten()
        mse_naive = np.mean((y_test_true_raw[SEQ_LENGTH:] - naive_pred)**2)
        results['Naive'].append(mse_naive)
        log(f"   ⚓ Naive Baseline MSE: {mse_naive:,.0f}")

        # Classical
        fold_scores_classical = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = LSTMModel(input_dim, HIDDEN).to(DEVICE)
            model = train_one_model(model, train_loader, val_loader, epochs=25, lr=LR)
            
            model.eval()
            with torch.no_grad():
                pred_scaled = model(test_tensor).cpu().numpy()
            pred = scaler.inverse_transform(pred_scaled).flatten()
            
            mse = np.mean((y_test_true_raw[SEQ_LENGTH:] - pred)**2)
            fold_scores_classical.append(mse)
            print(f"     Classical Seed {seed}: {mse:,.0f}")
            
        avg_c = np.mean(fold_scores_classical)
        results['Classical'].append(avg_c)
        log(f"   👉 Classical Avg: {avg_c:,.0f}")
        
        # Hybrid
        fold_scores_hybrid = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = HybridModel(input_dim, HIDDEN).to(DEVICE)
            model = train_one_model(model, train_loader, val_loader, epochs=5, lr=LR) 
            
            model.eval()
            with torch.no_grad():
                preds = []
                batch_start = 0
                while batch_start < len(test_tensor):
                    batch_end = batch_start + 512
                    batch_x = test_tensor[batch_start:batch_end]
                    preds.append(model(batch_x).cpu().numpy())
                    batch_start = batch_end
                pred_scaled = np.concatenate(preds)

            pred = scaler.inverse_transform(pred_scaled).flatten()
            
            mse = np.mean((y_test_true_raw[SEQ_LENGTH:] - pred)**2)
            fold_scores_hybrid.append(mse)
            print(f"     Hybrid Seed {seed}: {mse:,.0f}")
            
        avg_h = np.mean(fold_scores_hybrid)
        results['Hybrid'].append(avg_h)
        log(f"   👉 Hybrid Avg: {avg_h:,.0f}")

    log("\n" + "="*60)
    log("🏁 FINAL V1.5 BENCHMARK RESULTS")
    log("="*60)
    log(f"Hyperparameters: Hidden={HIDDEN}, LR={LR}")
    log(f"Naive:     {np.mean(results['Naive']):,.0f}")
    log(f"Classical: {np.mean(results['Classical']):,.0f}")
    log(f"Hybrid:    {np.mean(results['Hybrid']):,.0f}")

if __name__ == "__main__":
    run_v1_5()