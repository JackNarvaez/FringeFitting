import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from numpy.fft           import fftshift, fft2, fftfreq

# Path to the MeasurementSet (MS)
data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

# Open MS
data    = table(data_MS)

time    = np.unique(data.getcol("TIME"))
t0      = time[0]
time   -= t0
nt      = len(time)

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freq    = spectral_window[0]['CHAN_FREQ']
nf      = len(freq)
spectral_window.close()

antennas = table(data_MS+"/ANTENNA")
nant     = len(antennas.getcol("NAME"))
antennas.close()

nscale   = 8
n_pol    = 0
ant1     = 30
ant2     = 72

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
    exit()

t1  = data.query('ANTENNA1 == '+str(ant1) +' AND ANTENNA2 == '+str(ant2))   # do row selection

nnn = int(len(time)/8)-1
ff1 = 0
ff2 = len(freq)-1
tt1 = 0*nnn
tt2 = tt1+nnn

vis_n = t1.getcol('DATA')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
vis_new = np.exp(1j*np.angle(vis_n))
flg = t1.getcol('FLAG')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])

####### FFT ######
vis_new[flg] = 0.0

data.close()

####### FFT ######
nnt, nnf = vis_new.shape

dt = time[1] - time[0]       # time resolution [s]
df = freq[1] - freq[0]       # freq resolution [Hz]

F_s = fftshift(fft2(vis_new, norm="ortho", s=[nscale*nnt, nscale*nnf]))

fringe_rate = fftshift(fftfreq(nscale*nnt, dt))     # [Hz]
delay     = fftshift(fftfreq(nscale*nnf, df))*1e6 # [mu s]

power_FT  = np.abs(F_s)
phase_FT  = np.angle(F_s)
phase     = np.angle(vis_new)

peak_idx  = np.unravel_index(np.argmax(power_FT), power_FT.shape)
r_max_idx, tau_max_idx = peak_idx

phase_max = phase_FT[r_max_idx, tau_max_idx]
r_max     = fringe_rate[r_max_idx]  # Hz
tau_max   = delay[tau_max_idx]      # seconds

print(f"Estimated Fringe Rate (mHz): {r_max*1e3}")
print(f"Estimated Delay (ns):        {tau_max*1e3}")
print(f"Estimated Phase (rads):      {phase_max}")


plt.figure(figsize=(12, 8))
plt.title(f"Vis for Baseline {ant1}-{ant2}")
plt.imshow(phase.T, aspect='auto', origin='lower',
                    extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')

plt.axhline(freq[ff1]/1e6, ls="--",color="k")
plt.axhline(freq[ff2]/1e6, ls="--",color="k")
plt.axvline(time[tt1], ls="--", color="k")
plt.axvline(time[tt2], ls="--", color="k")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
plt.colorbar(label="Phase (rads)")
plt.show()

plt.figure(figsize=(8, 6))
plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
plt.imshow(power_FT.T, aspect='auto', origin='lower', norm="log",
           extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]],
           cmap='plasma')
plt.axvline(r_max, color='white', linestyle='--', linewidth=1)
plt.axhline(tau_max, color='white', linestyle='--', linewidth=1)
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"Delay ($\mu$s)")
plt.colorbar(label="Amplitude")
plt.show()


from scipy.ndimage import gaussian_filter1d

power_smoothed = gaussian_filter1d(power_FT[r_max_idx, :], sigma=1)
power1_smoothed = gaussian_filter1d(power_FT[:, tau_max_idx], sigma=1)

plt.plot(delay, power_FT[r_max_idx,:], ".", ms=2, c="k")
plt.plot(delay, power_smoothed, "-", c="b", alpha=0.6)
plt.xlabel(r"Delay ($\mu$s)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(delay[0], delay[-1])
plt.ylim(0)
plt.show()

plt.plot(fringe_rate, power_FT[:, tau_max_idx], ".", ms=2, c="k")
plt.plot(fringe_rate, power1_smoothed, "-", c="b", alpha=0.6)
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(fringe_rate[0], fringe_rate[-1])
plt.ylim(0)
plt.show()