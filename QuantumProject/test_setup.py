import torch
import pennylane as qml
import sys

print("="*30)
print("🚀 Checking System Status...")
print("="*30)

# 1. Check GPU for Deep Learning
print(f"🐍 Python Version: {sys.version.split()[0]}")
print(f"🔥 PyTorch Version: {torch.__version__}")
gpu_status = torch.cuda.is_available()
print(f"🖥️  GPU Available: {gpu_status}")

if gpu_status:
    print(f"🏆 GPU Name: {torch.cuda.get_device_name(0)}")
    print("   (Excellent! The RTX 3090 is ready to work.)")
else:
    print("❌ WARNING: GPU not detected!")

# 2. Check Quantum Library
print("-" * 20)
print(f"⚛️  PennyLane Version: {qml.__version__}")
dev = qml.device("default.qubit", wires=2)
print("✅ Quantum Device created successfully.")

print("="*30)
print("🎉 SYSTEM READY FOR BATTLE!")
print("="*30)