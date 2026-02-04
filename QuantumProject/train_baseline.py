import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

print("="*40)
print("🏋️  STARTING CLASSICAL BASELINE TRAINING")
print("="*40)

# 1. إعداد الجهاز (GPU Check)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")

# 2. تحميل البيانات المعالجة
print("📂 Loading processed data...")
data = torch.load('processed_data.pt')
X_train, y_train = data['X_train'].to(device), data['y_train'].to(device)
X_test, y_test = data['X_test'].to(device), data['y_test'].to(device)

# استخدام DataLoader للسرعة (Batching)
# Batch Size = 1024 (كبير لأن الذاكرة 24GB تسمح بذلك لتسريع التدريب)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1024, shuffle=False)

# 3. تصميم النموذج (Classical LSTM)
class ClassicalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(ClassicalLSTM, self).__init__()
        # طبقة LSTM تستلم التسلسل الزمني
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # طبقة خطية للتوقع النهائي
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # نأخذ آخر خطوة زمنية فقط (Many-to-One)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = ClassicalLSTM().to(device)
criterion = nn.MSELoss() # مقياس الخطأ (Mean Squared Error)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("-" * 30)
print("🏗️  Model Architecture Created (Standard LSTM)")
print(f"📊 Total Parameters: {sum(p.numel() for p in model.parameters())}")
print("-" * 30)

# 4. حلقة التدريب (Training Loop)
EPOCHS = 50
train_losses = []
test_losses = []

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch.unsqueeze(1)) # تعديل الشكل ليتطابق
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # التقييم (Validation)
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for X_val, y_val in test_loader:
            preds = model(X_val)
            loss = criterion(preds, y_val.unsqueeze(1))
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

total_time = time.time() - start_time
print("="*40)
print("🏁 TRAINING COMPLETE!")
print(f"⏱️  Total Time: {total_time:.2f} seconds")
print(f"📉 Final Test MSE: {test_losses[-1]:.6f}")
print("="*40)

# حفظ النتائج للمقارنة لاحقاً
torch.save(model.state_dict(), 'baseline_model.pth')
np.save('baseline_metrics.npy', {'train': train_losses, 'test': test_losses, 'time': total_time})