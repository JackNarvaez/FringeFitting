import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data (tab or multiple spaces as delimiter)
df = pd.read_csv("../Data/test.txt", delim_whitespace=True)

xd  = df["disp"]
xx  = np.linspace(xd[0], xd[len(df["disp"])-1], 100)
ydt = 0.5*(df["Sgmpt"] + df["Sgmmt"])
ydr = 0.5*(df["Sgmpr"] + df["Sgmmr"])

z = np.polyfit(xd, ydt, 2)
k_ext = np.sqrt(np.abs((ydt - z[1]*xd - z[2])/z[0]))
p = np.poly1d(z)

z1 = np.polyfit(xd, ydt, 4)
p1 = np.poly1d(z1)

z2 = np.polyfit(xd, ydt, 6)
p2 = np.poly1d(z2)

# Plot each column
plt.figure(figsize=(10,6))
plt.plot(xd[:70], ydt[:70]**2, marker='o', c="b", label="Delay", ms=3)
# plt.plot(xd, ydr, marker='o', c="g", label="Rate", ms=3)
# plt.plot(xx, p(xx), "--", c="y", label="2th", alpha=0.5)
# plt.plot(xx, p1(xx), "--", c="orange", label="4th", alpha=0.5)
# plt.plot(xx, p2(xx), "--", c="crimson", label="6th", alpha=0.5)
plt.loglog()

# Formatting
plt.xlabel("k_disp")
plt.ylabel("sigma")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.figure()
plt.plot(xd, k_ext - xd, marker='o', label="Noise OFF")
plt.xscale("log")
plt.show()

# Plot each column
plt.figure(figsize=(10,6))
plt.plot(xd[20:], ydt[20:], marker='o', c="b", label="Delay")
plt.loglog()
plt.xlabel("k_disp")
plt.ylabel("sigma")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

