## 📉 V1.5 Benchmark: The Information Bottleneck Discovery
In version 1.5, we implemented a **Strict Fairness Protocol** to evaluate the hybrid model against a highly tuned Classical LSTM.

### ⚙️ Protocol
* **Data:** 3 Rolling Windows (Summer, Fall, Winter).
* **Tuning:** Classical model was allowed to optimize hidden size (Winner: 128 units).
* **Fairness:** Identical seeds and data splits for both models.

### 📊 Final Results (MSE)
| Model | Avg MSE (Lower is Better) | Notes |
| :--- | :--- | :--- |
| **Naive Baseline** | 2,197,253 | Standard reference. |
| **Classical LSTM** | **64,652** 🏆 | Highly effective with 128 hidden units. |
| **Hybrid Quantum** | 132,360 | Struggled to compress data into 4 Qubits. |

### 🔬 Scientific Conclusion
The experiment revealed a critical **Information Bottleneck**. While the Classical model thrived with a wide hidden layer (128 neurons), the Hybrid model was forced to compress these 128 features into a narrow **4-Qubit Quantum Circuit**.
This compression caused significant information loss, proving that **simply adding a small quantum layer to a large classical network is insufficient.**

### 🔮 Next Step: V2.0
To overcome this bottleneck, Version 2.0 will introduce:
1.  **GPU Acceleration (cuQuantum):** To handle larger circuits.
2.  **Higher Qubit Count:** Increasing capacity from 4 to 8+ qubits.
3.  **Residual Architecture:** Allowing classical information to bypass the quantum bottleneck.