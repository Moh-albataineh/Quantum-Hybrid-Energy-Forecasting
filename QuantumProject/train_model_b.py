import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time

print("="*50)
print("🥊 V1.3 CHALLENGER: CLASSICAL LSTM + MLP (Model B)")
print("="*50)

# 1. إعدادات
SEQ_LENGTH = 168
BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. تحميل البيانات (نفس السابق)
print("📂 Loading Data...")
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

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device) # For batched evaluation
y_true = scaler.inverse_transform(y_test) # For final check

# 3. بناء الموديل المعزز (Model B)
class LSTM_MLP(nn.Module):
    def __init__(self):
        super(LSTM_MLP, self).__init__()
        # نفس الـ LSTM المستخدم في الكوانتوم
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        
        # إضافة MLP لمحاكاة الطبقات الكمومية كلاسيكياً
        # الـ Hybrid كان يضغط 50 -> 4 -> 1
        # نحن هنا سنفعل 50 -> 32 -> 16 -> 1 ليكون "عميقاً" وقوياً
        self.mlp = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last timestep
        out = self.mlp(out)
        return out

model = LSTM_MLP().to(device)

# حساب عدد الباراميترات للمقارنة
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model B Parameters: {total_params:,}")
# للمقارنة: الـ Hybrid كان يملك حوالي ~35,000 باراميتر تقريباً.
# هذا الموديل يجب أن يكون مقارباً أو أكبر قليلاً لنثبت أن الكوانتوم أكفأ.

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# 4. التدريب
print(" Training Model B (50 Epochs)...")
t0 = time.time()
model.train()
for epoch in range(50):
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        preds = model(X_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"   Epoch {epoch+1}/50 | Loss: {loss.item():.6f}")

print(f"⏱️  Time: {time.time() - t0:.2f}s")

# 5. التقييم النهائي
print("🔮 Evaluating Model B...")
test_loader = DataLoader(TensorDataset(X_test_t), batch_size=1024, shuffle=False)
model.eval()
preds_list = []
with torch.no_grad():
    for X_b in test_loader:
        p = model(X_b[0])
        preds_list.append(p.cpu().numpy())

pred_scaled = np.concatenate(preds_list)
pred_mw = scaler.inverse_transform(pred_scaled)

mse_model_b = np.mean((y_true - pred_mw)**2)

print("="*50)
print("🏆 RESULT: Model B (LSTM+MLP)")
print(f"   MSE: {mse_model_b:,.2f}")
print("-" * 30)

hybrid_score = 124751.44 # نتيجتنا السابقة
print(f"🆚 Hybrid Quantum Score: {hybrid_score:,.2f}")

if hybrid_score < mse_model_b:
    print("✅ SUCCESS: Quantum STILL WINS against the enhanced classical model!")
    print(f"   Improvement: {(mse_model_b/hybrid_score):.2f}x better")
else:
    print("❌ CHALLENGE: The extra MLP layers helped classical beat Quantum.")
    print("   This means the previous win was due to 'depth', not 'quantumness'.")