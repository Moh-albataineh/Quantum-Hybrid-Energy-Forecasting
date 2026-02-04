import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time

print("="*50)
print("📅 V1.2 BENCHMARK: THE WEEKLY WINDOW (FIXED MEMORY)")
print("="*50)

# 1. إعدادات
SEQ_LENGTH = 168  # أسبوع كامل
BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Hardware: {torch.cuda.get_device_name(0)}")

# 2. تحميل ومعالجة البيانات
print("📂 Loading data...")
df = pd.read_csv('PJME_hourly.csv', index_col='Datetime', parse_dates=True)
df.sort_index(inplace=True)
raw_data = df.values

train_size = int(len(raw_data) * 0.8)
train_data = raw_data[:train_size]
test_data = raw_data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# تحويل لـ Tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

print(f"✅ Data Processed. Test Shape: {X_test_t.shape}")

# 3. Baselines (Numpy calculations)
y_true = scaler.inverse_transform(y_test)

# Naive (t-1)
pred_naive = scaler.inverse_transform(X_test[:, -1, :])
# Daily (t-24)
pred_daily = scaler.inverse_transform(X_test[:, -24, :])
# Weekly (t-168)
pred_weekly = scaler.inverse_transform(X_test[:, 0, :])

def calc_mse(name, pred, true):
    mse = np.mean((true - pred)**2)
    print(f"📊 {name:20s} | MSE: {mse:10.2f}")
    return mse

print("-" * 30)
mse_naive = calc_mse("Naive (t-1)", pred_naive, y_true)
mse_daily = calc_mse("Daily (t-24)", pred_daily, y_true)
mse_weekly = calc_mse("Weekly (t-168)", pred_weekly, y_true)
print("-" * 30)

# 4. الموديل
class RobustLSTM(nn.Module):
    def __init__(self):
        super(RobustLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = RobustLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

print("🏋️  Training LSTM...")
model.train()
for epoch in range(30):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"   Epoch {epoch+1}/30 | Loss: {loss.item():.6f}")

# ==========================================
# 🛠️ الإصلاح هنا: التقييم على دفعات (Batched Evaluation)
# ==========================================
print("🔮 Evaluating in Batches (To save GPU RAM)...")
test_loader = DataLoader(TensorDataset(X_test_t), batch_size=BATCH_SIZE, shuffle=False)
model.eval()
predictions = []

with torch.no_grad():
    for X_batch in test_loader:
        # X_batch هو قائمة تحتوي عنصر واحد لأن TensorDataset فيه مدخل واحد
        batch_preds = model(X_batch[0]) 
        predictions.append(batch_preds.cpu().numpy())

# تجميع القطع
pred_lstm_scaled = np.concatenate(predictions)
pred_lstm = scaler.inverse_transform(pred_lstm_scaled)

mse_lstm = calc_mse("LSTM (Weekly)", pred_lstm, y_true)

print("="*50)
print("🏆 FINAL VERDICT (V1.2)")
print("="*50)
results = {
    "Naive (t-1)": mse_naive,
    "Daily (t-24)": mse_daily,
    "Weekly (t-168)": mse_weekly,
    "LSTM (Weekly)": mse_lstm
}
sorted_res = sorted(results.items(), key=lambda x: x[1])
for i, (k, v) in enumerate(sorted_res):
    print(f"{i+1}. {k}: {v:.2f}")

if sorted_res[0][0] == "LSTM (Weekly)":
    print("\n✅ VICTORY! LSTM Wins!")
else:
    print("\n❌ CHALLENGE: Baseline is still stronger.")