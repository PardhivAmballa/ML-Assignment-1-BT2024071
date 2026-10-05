"""
Master Execution Script for ML Assignment 1
"""

from src.train_var1 import train_phase1
from src.train_var2 import train_phase2

def main():
    print("==========================================================")
    print(" Running Complete Pipeline for ML Assignment 1")
    print("==========================================================\n")

    deg1, alpha1, cv_mse1, r2_1 = train_phase1()
    deg2, alpha2, cv_mse2, r2_2 = train_phase2()

    print("==========================================================")
    print("                   SUMMARY MATRIX                        ")
    print("==========================================================")
    print(f"{'Phase':<20} | {'Degree':<8} | {'Alpha':<8} | {'CV MSE':<10} | {'Train R2':<10}")
    print("-" * 65)
    print(f"{'Phase 1 (var1)':<20} | {deg1:<8} | {alpha1:<8} | {cv_mse1:<10.6f} | {r2_1:<10.6f}")
    print(f"{'Phase 2 (var2)':<20} | {deg2:<8} | {alpha2:<8} | {cv_mse2:<10.6f} | {r2_2:<10.6f}")
    print("==========================================================\n")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()