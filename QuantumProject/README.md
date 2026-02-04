# ⚛️ Hybrid Quantum-Classical LSTM for Energy Forecasting (V1.3)

## 🚀 Project Overview
This project benchmarks a **Hybrid VQC-LSTM** against Classical baselines for time-series forecasting.
Using a **168-hour (1 week) look-back window** on the PJME energy dataset, we conducted a **Multi-Seed Benchmark (5 runs)** to evaluate average performance and stability.

## 📊 Empirical Results (V1.3)
We compared three architectures over 5 random seeds [42, 101, 777, 2026, 99].

| Model Architecture | Avg Test MSE (MW²) | Best Test MSE | Stability (Std Dev) |
|--------------------|-------------------|---------------|---------------------|
| **Classical LSTM** | 211,093 | 193,075 | ± 13,004 (High Stability) |
| **Classical LSTM+MLP** | 340,812 | 218,124 | ± 193,392 (Unstable) |
| **Hybrid Quantum** | **177,647** | **137,514** | ± 45,491 (Best Peak Perf.) |

### 🔬 Analysis & Observations
1.  **Average Improvement:** The Hybrid model achieved a **~19% lower MSE on average** compared to the standard Classical LSTM.
2.  **Peak Performance:** The Quantum model demonstrated a higher potential ceiling, with its best run (137k) significantly outperforming the best Classical run (193k).
3.  **Stability Note:** While the Hybrid model had the best average, it showed higher variance across seeds compared to the pure LSTM. One seed (42) performed worse than the classical baseline, indicating sensitivity to initialization.
4.  **Classical Complexity:** The "LSTM+MLP" baseline exhibited high instability (large standard deviation), suggesting that simply adding classical parameters without careful tuning (e.g., gradient clipping) does not guarantee better performance compared to the VQC approach.

## 🛠️ Tech Stack
* **Hardware:** NVIDIA RTX 3090 (RunPod).
* **Quantum Backend:** PennyLane `lightning.qubit` (Simulated).
* **Evaluation:** Mean Squared Error (MSE) on real Megawatt values.

---
*Preliminary Research Results - V1.3*