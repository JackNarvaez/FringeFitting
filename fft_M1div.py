import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq

# Path to the MeasurementSet (MS)
data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data    = table(data_MS)
n_pol   = 0

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
ant1     = 10
ant2     = 60

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
    exit()

t1  = data.query('ANTENNA1 == '+str(ant1) +' AND ANTENNA2 == '+str(ant2))   # do row selection

ff1 = 0
ff2 = int(len(freq)/1)-1
phase_max = np.zeros([5, 16])
delay_max = np.zeros([5, 16])
rate_max  = np.zeros([5, 16])

for cc in range(5):
    pp = 4-cc
    intervals = 2**pp
    ttt = int(len(time)/intervals)-1
    for nnn in range(intervals):
        tt1 = nnn*ttt
        tt2 = tt1+ttt

        vis = t1.getcol('DATA')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
        flg = t1.getcol('FLAG')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])

        ####### FFT ######
        vis[flg] = 0.0

        nt_sub, nf_sub = vis.shape

        # 2D FFT
        F_sub = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf_sub]))
        power_sub = np.abs(F_sub)
        phase_sub = np.angle(F_sub)
        phase     = np.angle(vis)

        # ---------------------
        # FFT axes
        # ---------------------
        time_new = np.linspace(time[tt1], time[tt2], nt_sub)
        dt = (time_new[1] - time_new[0])             # time resolution [s]

        freq_new = np.linspace(freq[ff1], freq[ff2], nf_sub)
        df = (freq_new[1] - freq_new[0])           # freq resolution [Hz]
        
        fringe_rate = fftshift(fftfreq(nscale*nt_sub, dt))  # in Hz
        delay       = fftshift(fftfreq(nscale*nf_sub, df))*1e6  # in [mu s]

        # ---------------------
        # Find argmax
        # ---------------------
        peak_idx = np.unravel_index(np.argmax(power_sub), power_sub.shape)
        r_max_idx, tau_max_idx = peak_idx

        r_max = fringe_rate[r_max_idx]   # Hz
        tau_max = delay[tau_max_idx]     # seconds

        phase_max[pp, nnn] = phase_sub[r_max_idx, tau_max_idx]
        delay_max[pp, nnn] = tau_max*1e3
        rate_max[pp, nnn]  = r_max*1e3

data.close()

print("Phase:")
for pp in range(5):
    for nnn in range(2**pp):
        print(f"{phase_max[pp, nnn]:.3f}\t", end="")
    print("\n")

print("fringe rate:")
for pp in range(5):
    for nnn in range(2**pp):
        print(f"{rate_max[pp, nnn]:.3f}\t", end="")
    print("\n")

print("delay:")
for pp in range(5):
    for nnn in range(2**pp):
        print(f"{delay_max[pp, nnn]:.3f}\t", end="")
    print("\n")

f_idx =int(nf_sub/2)
t_idx = int(nt_sub/3)

# Create figure and grid layout
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
              wspace=0.05, hspace=0.05)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.plot(time_new, phase[:, f_idx], ".", color='purple')
ax_top.set_ylabel("Phase")
ax_top.set_xticks([])

ax_right = fig.add_subplot(gs[1, 1])
ax_right.plot(phase[t_idx, :], freq_new, ".", color='green')
ax_right.set_xlabel("Phase")
ax_right.set_yticks([])

ax_main = fig.add_subplot(gs[1, 0])

im = plt.imshow(phase.T, aspect='auto', origin='lower',
                    extent=[time[tt1], time[tt2], freq[ff1]/1e6, freq[ff2]/1e6],
                    cmap='twilight')
ax_main.axhline(freq[ff1+f_idx]/1e6, ls="--",color="k")
ax_main.axvline(time[tt1+t_idx], ls="--", color="k")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
cbar = fig.colorbar(im, ax=[ax_main, ax_top, ax_right], orientation='vertical',
                    pad=0.02, label="Phase (rads)")
plt.show()

plt.figure(figsize=(8, 6))
plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
plt.imshow(power_sub.T, aspect='auto', origin='lower', norm="log",
           extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]],
           cmap='plasma')
plt.axvline(r_max, color='white', linestyle='--', linewidth=1)
plt.axhline(tau_max, color='white', linestyle='--', linewidth=1)
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"Delay ($\mu$s)")
plt.colorbar(label="Amplitude")
plt.show()

plt.plot(delay, power_sub[r_max_idx,:], ".", ms=2, c="k")
for aa in delay_max[-1]:
    if np.abs(aa)>1e-4:
        plt.axvline(aa, ls="--", c="b")
plt.xlabel(r"Delay ($\mu$s)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(delay[0], delay[-1])
plt.ylim(0)
plt.show()

plt.plot(fringe_rate, power_sub[:, tau_max_idx], ".", ms=2, c="k")
for aa in rate_max[-1]:
    if np.abs(aa)>1e-4:
        plt.axvline(aa, ls="--", c="b")
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(fringe_rate[0], fringe_rate[-1])
plt.ylim(0)
plt.show()