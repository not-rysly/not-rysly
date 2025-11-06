# ============================================
# Part A: Iterative Methods for Linear Equations
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# System of equations:
# 10x1 - x2 + 2x3 = 6
# -x1 + 11x2 - x3 + 3x3 = 25  → Simplify to: -x1 + 11x2 - x3 = 25
# 2x1 - x2 + 10x3 = -11

# Coefficient matrix (A) and constants (b)
A = np.array([[10, -1, 2],
              [-1, 11, -1],
              [2, -1, 10]], dtype=float)
b = np.array([6, 25, -11], dtype=float)


# --- Jacobi Method ---
def jacobi(A, b, tol, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    errors = []
    for it in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        if error < tol:
            return x_new, it + 1, errors
        x[:] = x_new
    return x, max_iter, errors


# --- Gauss-Seidel Method ---
def gauss_seidel(A, b, tol, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    errors = []
    for it in range(max_iter):
        x_old = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        error = np.linalg.norm(x - x_old)
        errors.append(error)
        if error < tol:
            return x, it + 1, errors
    return x, max_iter, errors


# --- User input and results ---
tol = float(input("Enter tolerance (e.g. 1e-4): "))

x_jacobi, it_jacobi, err_jacobi = jacobi(A, b, tol)
x_gs, it_gs, err_gs = gauss_seidel(A, b, tol)

print("\n=== Jacobi Method ===")
print("Approximate solution:", x_jacobi)
print("Iterations:", it_jacobi)

print("\n=== Gauss-Seidel Method ===")
print("Approximate solution:", x_gs)
print("Iterations:", it_gs)

# --- Compare Performance ---
print("\nComparison:")
if it_jacobi > it_gs:
    print(f"Gauss-Seidel converged faster ({it_gs} iterations) than Jacobi ({it_jacobi} iterations).")
else:
    print(f"Jacobi converged faster ({it_jacobi} iterations) than Gauss-Seidel ({it_gs} iterations).")

# --- Bonus: Error convergence plot ---
plt.figure(figsize=(6, 4))
plt.semilogy(err_jacobi, label="Jacobi")
plt.semilogy(err_gs, label="Gauss-Seidel")
plt.xlabel("Iterations")
plt.ylabel("Error (log scale)")
plt.title("Error Convergence Comparison")
plt.legend()
plt.grid(True)
plt.show()
