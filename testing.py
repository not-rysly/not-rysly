import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.linspace(0, 10, 15)
y = 3*x**2 + 2*x + 5 + np.random.randn(15)*5  # noisy data

# NumPy polyfit
coeffs = np.polyfit(x, y, 2)
poly_eq = np.poly1d(coeffs)
y_fit_np = poly_eq(x)

# SciPy curve_fit
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

params, _ = curve_fit(quadratic, x, y)
y_fit_scipy = quadratic(x, *params)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res/ss_tot)

r2_np = r_squared(y, y_fit_np)
r2_scipy = r_squared(y, y_fit_scipy)

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_fit_np, 'r--', label=f'NumPy Fit (R²={r2_np:.3f})')
plt.plot(x, y_fit_scipy, 'g-', label=f'SciPy Fit (R²={r2_scipy:.3f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()

print("\nNumPy Fit: y = %.3fx² + %.3fx + %.3f" % tuple(coeffs))
print("R² =", round(r2_np, 4))

print("\nSciPy Fit: y = %.3fx² + %.3fx + %.3f" % tuple(params))
print("R² =", round(r2_scipy, 4))
