import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib  # لحفظ أداة التقييس لاستخدامها لاحقاً

print("="*40)
print("⚙️  STARTING DATA PROCESSING...")
print("="*40)

# 1. تحميل البيانات التي حفظناها سابقاً
df = pd.read_csv('PJME_hourly.csv', index_col='Datetime', parse_dates=True)
raw_data = df.values
print(f"📚 Original Data Loaded. Shape: {raw_data.shape}")

# 2. التقييس (Normalization) - ضغط البيانات بين 0 و 1
# هذه الخطوة جوهرية لتسريع تدريب الشبكات العصبية
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(raw_data)

# حفظ الـ Scaler لنستخدمه لاحقاً لعكس العملية (لمعرفة الأرقام الحقيقية)
joblib.dump(scaler, 'scaler.pkl')
print("📏 Data Normalized & Scaler saved.")

# 3. دالة النافذة المنزلقة (Sliding Window Function)
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        # نأخذ نافذة بحجم seq_length (الماضي)
        x = data[i:(i + seq_length)]
        # نأخذ القيمة التي تليها مباشرة (المستقبل)
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# سنستخدم نافذة بحجم 24 ساعة (يوم كامل) لتوقع الساعة القادمة
SEQ_LENGTH = 24
print(f"✂️  Slicing data into {SEQ_LENGTH}-hour sequences...")

X, y = create_sequences(data_normalized, SEQ_LENGTH)

# 4. التقسيم إلى تدريب واختبار (Train/Test Split)
# تحذير هام: لا نستخدم العشوائية (Shuffle) مع الزمن! يجب أن نحترم الترتيب.
train_size = int(len(X) * 0.8)  # 80% للتدريب

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# 5. التحويل إلى PyTorch Tensors (لأن كرت الشاشة يفهم Tensors فقط)
# نحتاج لتغيير الشكل ليكون: (Batch_Size, Sequence_Length, Features)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("-" * 30)
print("✅ PROCESSING COMPLETE!")
print(f"🔹 Training Data Shape: {X_train.shape}")
print(f"🔹 Testing Data Shape:  {X_test.shape}")
print("-" * 30)

# حفظ البيانات المعالجة لتسريع التحميل لاحقاً
torch.save({'X_train': X_train, 'y_train': y_train, 
            'X_test': X_test, 'y_test': y_test}, 'processed_data.pt')
print("💾 Processed tensors saved to 'processed_data.pt'")