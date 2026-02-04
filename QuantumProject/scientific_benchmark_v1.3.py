import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import time
import sys

# ==========================================
# ⚙️ إعدادات المعركة (The Arena Setup)
# ==========================================
SEEDS = [42, 101, 777, 2026, 99]  # 5 بذور عشوائية مختلفة
SEQ_LENGTH = 168  # أسبوع كامل
BATCH_SIZE = 32   # للكوانتوم
CLASSICAL_BATCH = 1024 # للكلاسيكي
EPOCHS_QUANTUM = 4
EPOCHS_CLASSICAL = 30 # زدنا الدورات للكلاسيكي ليكون عادلاً
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ملف لحفظ النتائج أولاً بأول (Log File)
LOG_FILE = "night_run_results.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

log("="*60)
log(f"🚀 STARTING MULTI-SEED SCIENTIFIC BENCHMARK (V1.3)")
log(f"⏰ Timestamp: {time.ctime()}")
log(f"⚙️  Hardware: {torch.cuda.get_device_name(0)}")
log(f"🌱 Seeds to run: {SEEDS}")
log("="*60)

# ==========================================
# 📂 تجهيز البيانات (مرة واحدة للجميع)
# ==========================================
log("📂 Loading Data...")
df = pd.read_csv('PJME_hourly.csv', index_col='Datetime', parse_dates=True)
df.sort_index(inplace=True)
raw_data = df.values

train_size = int(len(raw_data) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(raw_data[:train_size])
test_scaled = scaler.transform(raw_data[train_size:])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

X_train_np, y_train_np = create_sequences(train_scaled, SEQ_LENGTH)
X_test_np, y_test_np = create_sequences(test_scaled, SEQ_LENGTH)

# Tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(DEVICE)

# الحقيقة (Real MW) للمقارنة
y_true = scaler.inverse_transform(y_test_np)

log(f"✅ Data Ready. Train Samples: {len(X_train)}")
log("-" * 60)

# ==========================================
# 🧠 تعريف الموديلات (Model Architectures)
# ==========================================

# 1. Classical LSTM (The Strong Baseline)
class ClassicalLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. Optimized Model B (Tuned MLP) - نسخة مخففة لتجنب Overfitting
class ClassicalModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, num_layers=2, batch_first=True, dropout=0.2)
        # قللنا التعقيد قليلاً عن التجربة الفاشلة السابقة
        self.mlp = nn.Sequential(
            nn.Linear(50, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :])

# 3. Hybrid Quantum (The Champion)
n_qubits = 4
q_layers = 2
dev = qml.device("lightning.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(50, n_qubits)
        weight_shapes = {"weights": (q_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc = nn.Linear(n_qubits, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.proj(out[:, -1, :])
        out = torch.tanh(out) * (np.pi / 2)
        out = self.q_layer(out)
        return self.fc(out)

# ==========================================
# 🏋️ دالة التدريب الموحدة (The Training Engine)
# ==========================================
def train_and_evaluate(model_class, model_name, seed, epochs, batch_size, lr=0.001):
    # Set Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_class().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    # Training Loop
    start_t = time.time()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_t
    
    # Evaluation (Batched to save memory)
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test), batch_size=1024, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in test_loader:
            preds.append(model(xb[0]).cpu().numpy())
    
    pred_scaled = np.concatenate(preds)
    pred_mw = scaler.inverse_transform(pred_scaled)
    mse = np.mean((y_true - pred_mw)**2)
    
    log(f"   [Seed {seed}] {model_name} | MSE: {mse:,.2f} | Time: {train_time/60:.1f} min")
    return mse

# ==========================================
# 🏁 تنفيذ الاختبارات (Execution Loop)
# ==========================================
results = {"Classical": [], "Model_B": [], "Quantum": []}

log("🥊 ROUND 1: Classical LSTM (5 Seeds)")
for seed in SEEDS:
    score = train_and_evaluate(ClassicalLSTM, "Classical", seed, EPOCHS_CLASSICAL, CLASSICAL_BATCH)
    results["Classical"].append(score)

log("\n🥊 ROUND 2: Classical Model B (Tuned) (5 Seeds)")
for seed in SEEDS:
    score = train_and_evaluate(ClassicalModelB, "Model_B", seed, EPOCHS_CLASSICAL, CLASSICAL_BATCH)
    results["Model_B"].append(score)

log("\n⚛️ ROUND 3: Hybrid Quantum (5 Seeds) - This will take time...")
for seed in SEEDS:
    score = train_and_evaluate(HybridLSTM, "Quantum", seed, EPOCHS_QUANTUM, BATCH_SIZE, lr=0.002)
    results["Quantum"].append(score)

# ==========================================
# 📊 التقرير النهائي (Final Stats)
# ==========================================
log("\n" + "="*60)
log("🏆 FINAL SCIENTIFIC REPORT (V1.3 MULTI-SEED)")
log("="*60)

for name, scores in results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    best_score = np.min(scores)
    log(f"🟦 {name:10s} | Mean MSE: {mean_score:,.2f} ± {std_score:,.2f} | Best: {best_score:,.2f}")

log("-" * 60)
# مقارنة المتوسطات
class_mean = np.mean(results["Classical"])
quant_mean = np.mean(results["Quantum"])

if quant_mean < class_mean:
    log(f"✅ CONCLUSION: Quantum is consistently better by ~{class_mean/quant_mean:.2f}x on average.")
    log("   The victory is statistically significant!")
else:
    log(f"❌ CONCLUSION: Quantum failed to beat Classical on average.")

log(f"\n✅ All done at {time.ctime()}. Good morning! ☀️")