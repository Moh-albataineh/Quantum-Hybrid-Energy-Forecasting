import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
import time
import copy
import sys # For printing without newline
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# ⚙️ GLOBAL CONFIGURATION (V2.0: HEARTBEAT MONITOR)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_DEVICE_NAME = "lightning.gpu"

SEEDS = [42]
N_QUBITS = 8
HIDDEN_SIZE = 128
SEQ_LENGTH = 168
# 📉 Adjusted Batch Size for better feedback loop
BATCH_SIZE = 1024 
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 3

FOLDS = [
    {'train_end': 100000, 'val_size': 168*2, 'test_size': 168*2}, 
    {'train_end': 120000, 'val_size': 168*2, 'test_size': 168*2},
    {'train_end': 140000, 'val_size': 168*2, 'test_size': 168*2}
]

LOG_FILE = "v2.0_gpu_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ==========================================
# ⚛️ QUANTUM CIRCUIT
# ==========================================
try:
    dev = qml.device(Q_DEVICE_NAME, wires=N_QUBITS)
except qml.DeviceError:
    print("⚠️ lightning.gpu not found! Fallback to lightning.qubit.")
    dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ==========================================
# 🧠 MODELS
# ==========================================
class ClassicalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        features = out[:, -1, :]
        prediction = self.fc(features)
        return prediction, features

class QuantumResidualBlock(nn.Module):
    def __init__(self, hidden_dim, n_qubits):
        super().__init__()
        self.compressor = nn.Linear(hidden_dim, n_qubits) 
        weight_shapes = {"weights": (3, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post_process = nn.Linear(n_qubits, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        q_in = torch.tanh(self.compressor(features)) * np.pi 
        q_out = self.q_layer(q_in)
        residual = self.post_process(q_out)
        return residual * torch.tanh(self.alpha)

class ResQLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, n_qubits):
        super().__init__()
        self.classical = ClassicalLSTM(input_dim, hidden_size)
        self.quantum = QuantumResidualBlock(hidden_size, n_qubits)
        
    def forward(self, x, skip_quantum=False):
        c_pred, features = self.classical(x)
        if skip_quantum:
            return c_pred
        q_res = self.quantum(features)
        return c_pred + q_res

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
        return np.array(Xs), np.array(ys).reshape(-1, 1)
    X_train, y_train = scale_and_seq(train_raw, scaler_load)
    X_val, y_val = scale_and_seq(val_raw, scaler_load)
    X_test, y_test = scale_and_seq(test_raw, scaler_load)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_load, test_raw[:, 0]

def train_stage(model, stage_name, train_loader, val_loader, epochs, lr, freeze_classical=False):
    skip_q = not freeze_classical 
    
    if freeze_classical:
        for param in model.classical.parameters(): param.requires_grad = False
        optimizer = optim.AdamW(model.quantum.parameters(), lr=lr)
        log(f"   ❄️ Stage 2: Classical FROZEN. Quantum Active.")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        log(f"   🔥 Stage 1: Classical Warmup.")
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    best_weights = None
    
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0
        
        # 💓 V2.0.3 Feature: Real-time Step Monitoring
        steps = len(train_loader)
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb, skip_quantum=skip_q)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
            # Print progress every 10% of the epoch if in Quantum Stage
            if not skip_q and (i % 5 == 0 or i == steps-1):
                # Write to stdout directly to see "movement"
                sys.stdout.write(f"\r     >> [Ep {ep+1}] Step {i+1}/{steps} | Loss: {loss.item():.6f}")
                sys.stdout.flush()
        
        if not skip_q: print() # Newline after quantum epoch
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb, skip_quantum=skip_q)
                val_loss += criterion(pred, yb).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        alpha_val = model.quantum.alpha.item()
        dt = time.time() - t0
        log(f"     [{stage_name}] Ep {ep+1}/{epochs} | Val: {avg_val_loss:.6f} | Alpha: {alpha_val:.4f} | Time: {dt:.1f}s")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_weights)
    return model, best_loss

# ==========================================
# 🚀 MAIN
# ==========================================
def run_v2():
    log("🚀 STARTING V2.0 (HEARTBEAT): Batch=1024, Real-time Logs")
    log("="*60)
    df = pd.read_csv('PJME_hourly.csv', index_col='Datetime', parse_dates=True)
    df.sort_index(inplace=True)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    features_raw = df[['PJME_MW', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values.astype(float)
    
    results = {'Classical': [], 'Res_Quantum': []}
    
    for fold_idx, fold_cfg in enumerate(FOLDS):
        log(f"\n📅 FOLD {fold_idx+1}/{len(FOLDS)}")
        log("-" * 40)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), scaler, y_test_true_raw = get_data_split(features_raw, fold_cfg)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=BATCH_SIZE*4)
        test_tensor = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        
        fold_res_c = []
        fold_res_q = []
        
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            log(f"\n🌱 Seed {seed}:")
            model = ResQLSTM(5, HIDDEN_SIZE, N_QUBITS).to(DEVICE)
            
            # STAGE 1
            start_t = time.time()
            model, loss_c = train_stage(model, "Stage1-Classical", train_loader, val_loader, EPOCHS_STAGE1, lr=0.001, freeze_classical=False)
            
            model.eval()
            with torch.no_grad():
                pred_c_scaled, _ = model.classical(test_tensor)
                pred_c = scaler.inverse_transform(pred_c_scaled.cpu().numpy()).flatten()
            mse_c = np.mean((y_test_true_raw[SEQ_LENGTH:] - pred_c)**2)
            fold_res_c.append(mse_c)
            log(f"   👉 Stage 1 MSE: {mse_c:,.0f} | Time: {(time.time()-start_t)/60:.1f}m")
            
            # STAGE 2
            model, loss_q = train_stage(model, "Stage2-Quantum", train_loader, val_loader, EPOCHS_STAGE2, lr=0.005, freeze_classical=True)
            
            model.eval()
            with torch.no_grad():
                preds = []
                bs = 1024
                for i in range(0, len(test_tensor), bs):
                    preds.append(model(test_tensor[i:i+bs], skip_quantum=False).cpu().numpy())
                pred_q_scaled = np.concatenate(preds)
            pred_q = scaler.inverse_transform(pred_q_scaled).flatten()
            mse_q = np.mean((y_test_true_raw[SEQ_LENGTH:] - pred_q)**2)
            fold_res_q.append(mse_q)
            
            final_alpha = torch.tanh(model.quantum.alpha).item()
            log(f"   🚀 Stage 2 MSE: {mse_q:,.0f} | Alpha: {final_alpha:.4f}")
            
        avg_c = np.mean(fold_res_c)
        avg_q = np.mean(fold_res_q)
        results['Classical'].append(avg_c)
        results['Res_Quantum'].append(avg_q)
        log(f"📊 FOLD RESULT: Classical {avg_c:,.0f} vs Quantum {avg_q:,.0f}")

    log("\n🏁 FINAL V2.0 BENCHMARK RESULTS")
    log(f"Classical: {np.mean(results['Classical']):,.0f}")
    log(f"Quantum:   {np.mean(results['Res_Quantum']):,.0f}")

if __name__ == "__main__":
    run_v2()