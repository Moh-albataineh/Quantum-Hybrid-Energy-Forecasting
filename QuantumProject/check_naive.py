import torch
import numpy as np

print("="*40)
print("🧐 REALITY CHECK: Naive Persistence Model")
print("="*40)

# 1. تحميل بيانات الاختبار
data = torch.load('processed_data.pt')
# البيانات الحقيقية (المستهدف)
y_test = data['y_test'].numpy()
# المدخلات (آخر 24 ساعة)
X_test = data['X_test'].numpy()

# 2. بناء النموذج الساذج
# "توقع الساعة القادمة هو نفس قيمة الساعة الحالية (آخر قيمة في النافذة)"
# الشكل هو (Samples, Sequence_Length, Features) -> نأخذ (:, -1, :)
y_pred_naive = X_test[:, -1, :]

# 3. حساب نسبة الخطأ (MSE)
mse_naive = np.mean((y_test - y_pred_naive)**2)
print(f"📉 Naive Baseline MSE: {mse_naive:.6f}")

# 4. مقارنة مع رقمك الكمومي
your_quantum_mse = 0.000157 

print("-" * 30)
if your_quantum_mse < mse_naive:
    print(f"✅ EXCELLENT! Quantum Model ({your_quantum_mse}) is better than Naive ({mse_naive:.6f}).")
    print("   Result: The model is learning patterns, not just copying the last value.")
else:
    print(f"❌ WARNING: Quantum Model ({your_quantum_mse}) is WORSE/EQUAL to Naive ({mse_naive:.6f}).")
    print("   Result: The model is likely just mimicking the previous time step.")
print("="*40)