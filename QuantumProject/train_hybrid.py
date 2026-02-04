import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
import time
import numpy as np

print("="*40)
print("⚛️  STARTING HYBRID QUANTUM TRAINING (OPTIMIZED)")
print("="*40)

# 1. إعداد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")

# 2. تحميل البيانات
data = torch.load('processed_data.pt')
X_train, y_train = data['X_train'].to(device), data['y_train'].to(device)
X_test, y_test = data['X_test'].to(device), data['y_test'].to(device)

# تعديل هام 1: تقليل حجم الباتش ليسرع الكوانتوم
BATCH_SIZE = 32 
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 3. تصميم الدائرة الكمومية
n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits) # استخدام محاكي سريع

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 4. بناء النموذج الهجين
class HybridLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, n_qubits=4, q_layers=2):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.projector = nn.Linear(hidden_size, n_qubits)
        weight_shapes = {"weights": (q_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc_out = nn.Linear(n_qubits, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.projector(out)
        out = torch.tanh(out) * (np.pi / 2)
        out = self.quantum_layer(out)
        out = self.fc_out(out)
        return out

model = HybridLSTM(n_qubits=n_qubits).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # زدنا السرعة قليلاً

print("-" * 30)
print(f"🧠 Hybrid Model Created (Batch Size: {BATCH_SIZE})")
print("-" * 30)

# 5. حلقة التدريب
EPOCHS = 5 # سنبدأ بـ 5 حقب فقط للتجربة السريعة
train_losses = []
test_losses = []

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # إضافة شريط تقدم (Progress Bar) يدوي
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(X_batch)
        
        # تعديل هام 2: إصلاح تحذير الأبعاد (حذف unsqueeze)
        loss = criterion(predictions, y_batch) 
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 0:
            print(f"  Epoch {epoch+1} > Batch {i}/{len(train_loader)} | Loss: {loss.item():.6f}", end='\r')
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # التقييم
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for X_val, y_val in test_loader:
            preds = model(X_val)
            loss = criterion(preds, y_val) # إصلاح هنا أيضاً
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

total_time = time.time() - start_time
print("="*40)
print(f"🏁 HYBRID DONE! Time: {total_time:.2f}s")
print(f"📉 Final Test MSE: {test_losses[-1]:.6f}")
print("="*40)

torch.save(model.state_dict(), 'hybrid_model.pth')