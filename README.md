# Machine Learning Assignment 1: Polynomial Regression

**Student Name:** Amballa Pardhiv
**Roll Number:** BT2024071
**Repository:** https://github.com/PardhivAmballa/ML-Assignment-1-BT2024071

---

## 📌 Project Overview
This repository contains the full implementation for **Assignment 1: Polynomial Regression**. The objective is to model non-linear phenomena across two personalized datasets:
1. **Phase 1 (`var1`):** Steam Turbine Optimization (6 operational input features).
2. **Phase 2 (`var2`):** Subterranean Thermal Reservoir Mapping (3D spatial offsets).

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── BT2024071_train_var1.csv
│   ├── BT2024071_test_var1.csv
│   ├── BT2024071_train_var2.csv
│   └── BT2024071_test_var2.csv
├── predictions/
│   ├── BT2024071_pred_var1.csv
│   └── BT2024071_pred_var2.csv
├── plots/
│   ├── var1_cv_mse.png
│   └── var2_cv_mse.png
├── src/
│   ├── train_var1.py
│   └── train_var2.py
├── run_all.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/PardhivAmballa/ML-Assignment-1-BT2024071.git](https://github.com/PardhivAmballa/ML-Assignment-1-BT2024071.git)
   cd ML-Assignment-1-BT2024071
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution

To run the full cross-validation grid search, model training, plot generation, and test prediction pipeline:

```bash
python run_all.py
```

---

## 📊 Summary Results

| Problem Phase | Features ($n$) | Selected Degree ($d^*$) | Regularization ($\alpha^*$) | 5-Fold CV MSE | Train $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 (`var1`)** | 6 | **5** | **10.0** (Ridge) | **0.465630** | **0.983580** |
| **Phase 2 (`var2`)** | 3 | **11** | **1.0** (Ridge) | **0.243528** | **0.996307** |