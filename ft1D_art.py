import numpy             as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from numpy.fft           import fftshift, fft, fftfreq

data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data    = table(data_MS)
timei   = np.unique(data.getcol("TIME"))
t0, tf  = np.min(timei), np.max(timei)
nt      = len(timei)
time    = np.linspace(0, tf-t0, nt)
data.close()

nscale  = 8
phase   = np.zeros(nt)
r       = 1e-3  # Hz

phi0    = np.pi

noise = 0
for ii in range(nt):
    phase[ii] = np.angle(np.exp(1j*(phi0 + 2*np.pi*r*time[ii] + noise*np.random.normal(0,1,1))))

#1D FFT
F_ = fftshift(fft(np.exp(1j*phase), norm='ortho', n=nscale*nt))
ampl_  = np.abs(F_)
phase_ = np.angle(F_)

dt = (time[1] - time[0])             # time resolution [s]
fringe_rate = fftshift(fftfreq(nscale*nt, dt))  # in Hz

# ---------------------
# Find argmax
# ---------------------
r_max_idx = np.unravel_index(np.argmax(ampl_), ampl_.shape)
r_max = fringe_rate[r_max_idx]
phase_max = phase_[r_max_idx]

print(f"Phase:\t Teo: {phi0:.5f}\t Est: {phase_max:.5f}")
print(f"Rate :\t Teo: {r:.5f}\t Est: {r_max:.5f}")

plt.plot(time, phase, ".", ms=2, c="k")
plt.xlabel(r"time [s]")
plt.ylabel(r"Phase")
plt.ylim(-np.pi, np.pi)
plt.xlim(time[0], time[-1])
plt.show()

plt.plot(fringe_rate, ampl_, ".", ms=2, c="k")
plt.xlabel(r"Fringe Rate (Hz)")
plt.ylabel(r"$Arg\{\hat{F}\}$")
plt.xlim(fringe_rate[0], fringe_rate[-1])
plt.show()

plt.plot(fringe_rate, phase_, ".", ms=2, c="k")
plt.xlabel(r"Fringe Rate (Hz)")
plt.ylabel(r"$|\hat{F}|$")
plt.ylim(-np.pi, np.pi)
plt.xlim(fringe_rate[0], fringe_rate[-1])
plt.show()