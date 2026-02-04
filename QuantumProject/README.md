# ⚛️ Hybrid Quantum-Classical LSTM Project

## 🚀 Overview
An experimental study comparing Quantum Hybrid LSTM vs. Classical Baselines for energy forecasting.

## 📊 Key Results (The Truth)
| Model | Test MSE | Performance vs Naive |
|-------|----------|----------------------|
| **Naive Persistence** | **0.000838** | **1x (Baseline)** |
| Classical LSTM | ~0.0167 | Underperformed (Needs hyperparameter tuning) |
| **Hybrid Quantum** | **~0.00016** | **~5.3x Improvement over Naive** |

### 💡 Conclusion
The Hybrid Quantum model successfully learned temporal patterns, significantly outperforming the Naive benchmark (5x accuracy). The Classical LSTM baseline underperformed in this specific run, indicating strictly that the Quantum architecture converged faster/better with these specific hyperparameters, not necessarily that it is universally superior without further optimization.