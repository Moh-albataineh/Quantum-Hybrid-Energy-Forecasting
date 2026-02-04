import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import joblib

print("="*40)
print("🎨 GENERATING FINAL BATTLE CHART")
print("="*40)

# 1. إعدادات النموذج (يجب أن تطابق ما دربناه)
device = torch.device("cpu") # للرسم نستخدم CPU أسهل

# تعريف الكلاسات مرة أخرى لتحميل الأوزان
class ClassicalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(ClassicalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

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

# 2. تحميل البيانات والمقياس (Scaler)
print("📂 Loading data...")
data = torch.load('processed_data.pt')
scaler = joblib.load('scaler.pkl')

X_test = data['X_test'].to(device)
y_test = data['y_test'].to(device)

# نأخذ عينة صغيرة للرسم (أول 300 ساعة مثلاً) للوضوح
sample_size = 300
X_sample = X_test[:sample_size]
y_real_scaled = y_test[:sample_size].numpy()

# 3. تحميل النماذج وتشغيل التوقعات
print("🤖 Loading Classical Model...")
model_c = ClassicalLSTM().to(device)
model_c.load_state_dict(torch.load('baseline_model.pth', map_location=device))
model_c.eval()
with torch.no_grad():
    pred_c_scaled = model_c(X_sample).numpy()

print("⚛️ Loading Hybrid Quantum Model...")
model_q = HybridLSTM().to(device)
model_q.load_state_dict(torch.load('hybrid_model.pth', map_location=device))
model_q.eval()
with torch.no_grad():
    pred_q_scaled = model_q(X_sample).numpy()

# 4. عكس التقييس (للعودة للوحدة الحقيقية: ميجاواط)
print("🔄 Inverse Scaling (Calculating Real MW)...")
real_vals = scaler.inverse_transform(y_real_scaled.reshape(-1, 1))
pred_c_vals = scaler.inverse_transform(pred_c_scaled)
pred_q_vals = scaler.inverse_transform(pred_q_scaled)

# 5. الرسم
print("🖌️  Plotting comparison...")
plt.figure(figsize=(16, 8))
plt.plot(real_vals, label='Actual Energy Consumption (Real)', color='black', linewidth=2, alpha=0.7)
plt.plot(pred_c_vals, label='Classical AI Prediction', color='blue', linestyle='--', alpha=0.8)
plt.plot(pred_q_vals, label='Quantum Hybrid Prediction', color='red', linewidth=2, alpha=0.9)

plt.title('Final Battle: Classical AI vs Quantum Hybrid AI (PJM East Region)', fontsize=16)
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('Energy Consumption (MW)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plot_file = "final_victory_chart.png"
plt.savefig(plot_file)
print(f"✅ Chart saved as '{plot_file}'")