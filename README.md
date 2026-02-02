🚀 XGBoost-Based Anomaly Detection Pipeline

A production-ready machine learning pipeline for highly imbalanced binary classification problems (cybersecurity, fraud detection, network intrusion detection).  
This project focuses on **maximizing F1-score** using smart undersampling, feature engineering, and threshold optimization.

---

🔍 Features
- Smart majority-class undersampling
- Automatic datetime feature handling
- Categorical feature support
- XGBoost with regularization & histogram optimization
- Threshold tuning for maximum F1-score
- End-to-end pipeline: train → validate → retrain → test → submission CSV

---

🧠 Model Strategy
- **Class imbalance handling:** Custom undersampling strategy
- **Evaluation metric:** F1-score (optimized via threshold scanning)
- **Final training:** Retrains on full dataset for maximum generalization
- **Inference:** Probability-based classification

---

🛠 Tech Stack
- Python 3.9+
- XGBoost
- Pandas
- NumPy
- Scikit-learn

---


