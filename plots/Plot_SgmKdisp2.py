import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def fitted_func(x, A, B, C):
    return A + B*np.abs(x) + np.exp(-C*np.abs(x))

# Read the data (tab or multiple spaces as delimiter)
df = pd.read_csv( "../Data/sigma_disp_20_26.txt", delim_whitespace=True)
dfN = pd.read_csv("../Data/sigma_disp_22_26.txt", delim_whitespace=True)
dfM = pd.read_csv("../Data/sigma_disp_25_26.txt", delim_whitespace=True)
dfO = pd.read_csv("../Data/sigma_disp_18_26.txt", delim_whitespace=True)
df1 = pd.read_csv("../Data/LSM_disp.txt", delim_whitespace=True)
df2 = pd.read_csv("../Data/LSM_.txt", delim_whitespace=True)

xd  = df["k_disp"]
xdN = dfN["k_disp"]
xdM = dfM["k_disp"]
xdO = dfO["k_disp"]
xd1 = df1["k_disp"]
xd2 = df2["k_disp"]

X_data = np.concatenate((xd, xdN, xdM, xdO))

xx  = np.linspace(np.min(xd), np.max(xd), 100)
ydt = df["sigma"]
ydtN = dfN["sigma"]
ydtM = dfM["sigma"]
ydtO = dfO["sigma"]

Y_data = np.concatenate((ydt, ydtN, ydtM, ydtO))

z = np.polyfit(xd, ydt, 1)
# k_ext = np.sqrt(np.abs((ydt - z[1]*xd - z[2])/z[0]))
p = np.poly1d(z)

z1 = np.polyfit(xd, ydt, 2)
p1 = np.poly1d(z1)

# z2 = np.polyfit(xd, ydt, 6)
# p2 = np.poly1d(z2)

# Plot each column
plt.figure(figsize=(10,6))
# plt.plot(xd , ydt ,'.', c="k", label="20-26", ms=5)
# plt.plot(xdN, ydtN, '.', c="r", label="22-26", ms=5)
# plt.plot(xdM, ydtM, '.', c="b", label="25-26", ms=5)
# plt.plot(xdO, ydtO, '.', c="g", label="18-26", ms=5)
plt.plot(X_data, Y_data, '.', c="g", label="18-26", ms=5)
# plt.plot(xd, ydr, marker='o', c="g", label="Rate", ms=3)
# plt.plot(xx, p(xx), "--", c="crimson", label="1th", alpha=0.5)
# plt.plot(xx, p1(xx), "--", c="orange", label="2th", alpha=0.5)
# plt.plot(xx, p2(xx), "--", c="crimson", label="6th", alpha=0.5)
# plt.loglog()

popt, _ = curve_fit(fitted_func, X_data, Y_data, p0=[np.min(Y_data)-1, 1, 1])

plt.plot(X_data, fitted_func(X_data, popt[0], popt[1], popt[2]), '.', c="k", ms=5)
plt.ylim(0, np.max(Y_data))
# # Formatting
plt.xlabel(r"$k_{disp}$")
plt.ylabel(r"$\sigma$")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(xd, xd1, '.', label="LSM -> k_disp")
plt.plot(xd, xd2, '.', label="LSM -> 4 prms")
# plt.xscale("log")
# plt.show()

# # Plot each column
# plt.figure(figsize=(10,6))
# plt.plot(xd[20:], ydt[20:], marker='o', c="b", label="Delay")
# plt.loglog()
plt.xlabel("k_disp (emcee)")
plt.ylabel("k_disp")
plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)
plt.show()