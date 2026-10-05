"""
Phase 1: Steam Turbine Optimization (var1)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROLL_NO = "BT2024071"

def train_phase1(data_dir="data", output_dir="predictions", plot_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train_path = os.path.join(data_dir, f"{ROLL_NO}_train_var1.csv")
    test_path = os.path.join(data_dir, f"{ROLL_NO}_test_var1.csv")

    print(f"--- Phase 1 (var1): Loading {train_path} ---")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_test = test_df.drop(columns=['y']) if 'y' in test_df.columns else test_df

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    degrees = range(1, 7)
    alphas = [0.0, 0.1, 1.0, 10.0, 50.0]

    best_cv_mse = float('inf')
    best_deg = 1
    best_alpha = 0.0
    
    degree_cv_results = {}

    for d in degrees:
        degree_cv_results[d] = {}
        for a in alphas:
            mses = []
            for tr_idx, val_idx in kf.split(X_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                model = LinearRegression() if a == 0.0 else Ridge(alpha=a)
                pipe = Pipeline([
                    ('poly', PolynomialFeatures(degree=d, include_bias=True)),
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_val)
                mses.append(mean_squared_error(y_val, preds))

            avg_mse = float(np.mean(mses))
            degree_cv_results[d][a] = avg_mse

            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_deg = d
                best_alpha = a

    print(f"[Phase 1] Optimal Degree: {best_deg} | Optimal Alpha: {best_alpha} | Best CV MSE: {best_cv_mse:.6f}")

    # Plotting CV MSE vs Degree
    plt.figure(figsize=(8, 5))
    plot_mses = [degree_cv_results[d][best_alpha] for d in degrees]
    plt.plot(degrees, plot_mses, marker='o', linewidth=2, color='navy', label=f'Ridge (alpha={best_alpha})')
    plt.title('Phase 1 (var1): Validation MSE vs. Polynomial Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('5-Fold Cross-Validation MSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plot_file = os.path.join(plot_dir, "var1_cv_mse.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Phase 1] Saved CV plot to {plot_file}")

    # Fit final pipeline on full training data
    final_model = LinearRegression() if best_alpha == 0.0 else Ridge(alpha=best_alpha)
    final_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=best_deg, include_bias=True)),
        ('scaler', StandardScaler()),
        ('model', final_model)
    ])
    final_pipe.fit(X_train, y_train)

    train_preds = final_pipe.predict(X_train)
    full_mse = mean_squared_error(y_train, train_preds)
    full_r2 = r2_score(y_train, train_preds)
    print(f"[Phase 1] Full Train Set Performance -> MSE: {full_mse:.6f} | R2: {full_r2:.6f}")

    # Generate Test Predictions
    test_preds = final_pipe.predict(X_test)
    pred_df = pd.DataFrame({'y': test_preds})
    
    # Save predictions
    pred_filename = f"{ROLL_NO}_pred_var1.csv"
    pred_df.to_csv(os.path.join(output_dir, pred_filename), index=False)
    pred_df.to_csv(pred_filename, index=False)
    print(f"[Phase 1] Saved prediction file: {pred_filename}\n")

    return best_deg, best_alpha, best_cv_mse, full_r2

if __name__ == "__main__":
    train_phase1()