# ⚛️ Quantum-Hybrid Energy Forecasting (V2.0)

## 📌 Project Overview
This project benchmarks Hybrid Quantum-Classical Neural Networks for time-series forecasting (PJM Energy Data).
**Current Status:** V2.0 (Res-Q-LSTM) introduces a residual hybrid design intended to mitigate the information bottleneck observed in V1.5. In a single-seed pilot, it improved test MSE across 3 rolling folds under a fixed protocol.

## 🏆 V2.0: The Res-Q-LSTM Architecture
In this version, we transitioned to a **Residual Learning** framework accelerated by **NVIDIA cuQuantum**.

### 🔬 Methodology (Two-Stage Protocol)
1.  **Stage 1 (Classical Baseline):** Train a Classical LSTM (128 units) to predict the target.
2.  **Stage 2 (Residual Hybrid):** Freeze the classical backbone. Train **only** the Quantum Circuit (8 Qubits) + Projection Layer to predict the residual error:
    $$r = y_{true} - \hat{y}_{classical}$$
3.  **Final Prediction:**
    $$\hat{y} = \hat{y}_{classical} + \tanh(\alpha) \cdot \hat{r}_{quantum}$$
    *(Where $\alpha$ is a learnable gating parameter initialized at 0)*.

### ⚙️ Hardware & Backend
* **Backend:** PennyLane `lightning.gpu` (State-vector simulation).
* **Differentiation:** Adjoint Differentiation (Memory-efficient).
* **Hardware:** NVIDIA RTX 3090.

---

## 📊 V2.0 Pilot Benchmark Results
**Setup:** 3-Fold Rolling Window | 8 Qubits | Seed 42 | GPU Acceleration.

| Fold | Classical Baseline (MSE) | **Res-Q-LSTM (MSE)** | **Improvement** | **Gate ($\alpha$)** |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 1** | 289,760 | **230,197** | **+20.6%** | -0.0516 |
| **Fold 2** | 148,512 | **134,005** | **+9.8%** | +0.0324 |
| **Fold 3** | 231,381 | **228,650** | **+1.2%** | +0.0293 |
| **AVG** | **223,218** | **197,617** | **+11.5%** | **Non-Zero** |

### 💡 Interpretation & Limitations
* **Consistency:** The residual hybrid improved test MSE in all 3 folds in this pilot run.
* **Gate Behavior:** $\alpha$ consistently moved away from zero, indicating the optimization utilized the residual branch (gradient flow confirmed).
* **Scope:** This is a single-seed pilot due to high simulation cost (~12 hours). Full robustness will be assessed with additional seeds and a parameter-matched classical residual control.

---

## 🛠️ Reproducibility
To replicate the V2.0 benchmark: 

### 1. Prerequisites 
* Python 3.10+
* NVIDIA GPU (CUDA 12 support required for `cuQuantum`).

### 2. Setup
bash 
` pip install -r requirements.txt `



### 3. Data Placement
Ensure the dataset is in the root directory:

Place PJME_hourly.csv in the main project folder.

### 4. Run Benchmark
Bash
python scientific_benchmark_v2.py

---

## 📂 Outputs

Logs: v2.0_pilot_log.txt (Training progress and metrics).

Results: Console output showing per-fold MSE and Final Summary.

