import numpy             as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq

data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data    = table(data_MS)
n_rows  = int(len(data))

timeo   = np.unique(data.getcol("TIME"))
t0      = timeo[0]
timeo  -= t0
nt      = len(timeo)

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freqo   = spectral_window[0]['CHAN_FREQ']
nf      = len(freqo)
spectral_window.close()

data.close()

nscale  = 8

time    = np.linspace(timeo[0], timeo[-1], nt)
freq    = np.linspace(freqo[0], freqo[-1], nf)

phase_gen = np.zeros([nt, nf])

r       = 1e-3  # Hz 1 ms
tau     = 1e-7  # s  100 ns
phi0    = 0.5
noise   = 1
for ii in range(nt):
    for jj in range(nf):
        phase_gen[ii, jj] = np.angle(np.exp(1j*(phi0 + 2*np.pi*r*time[ii] + 2*np.pi*tau*(freq[jj]-freq[0]) + noise*np.random.normal(0,1,1))))

ff1 = 0
ff2 = int(len(freq)/1)-1
phase_max = np.zeros([5, 16])
delay_max = np.zeros([5, 16])
rate_max = np.zeros([5, 16])

for cc in range(5):
    pp = 4-cc
    intervals = 2**pp
    ttt = int(len(time)/intervals)-1
    for nnn in range(intervals):
        tt1 = nnn*ttt
        tt2 = tt1+ttt

        vis = phase_gen[tt1:tt2+1, ff1:ff2+1]   # (channel, pol [RR, RL, LR, LL])

        nt_sub, nf_sub = vis.shape

        # 2D FFT
        F_sub = fftshift(fft2(np.exp(1j*vis), norm='ortho', s=[nscale*nt_sub, nscale*nf_sub]))
        power_sub = np.abs(F_sub)
        phase_sub = np.angle(F_sub)
        phase     = vis

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
        #flat_indices = np.argsort(power_sub, axis=None)
        #second_peak_flat_idx = flat_indices[-1]
        #peak_idx = np.unravel_index(second_peak_flat_idx, power_sub.shape)
        r_max_idx, tau_max_idx = peak_idx

        r_max = fringe_rate[r_max_idx]   # Hz
        tau_max = delay[tau_max_idx]     # seconds

        phase_max[pp, nnn] = phase_sub[r_max_idx, tau_max_idx]
        delay_max[pp, nnn] = tau_max*1e3
        rate_max[pp, nnn]  = r_max*1e3

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

im = plt.imshow(phase.T, aspect='auto', origin='lower', vmin=-np.pi, vmax=np.pi,
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
plt.imshow(power_sub.T, aspect='auto', origin='lower', norm="log",
           extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]],
           cmap='plasma')

plt.axvline(r_max, color='white', linestyle='--', linewidth=1)
plt.axhline(tau_max, color='white', linestyle='--', linewidth=1)
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"Delay ($\mu$s)")
plt.colorbar(label="Amplitude")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(phase_sub.T, aspect='auto', origin='lower', vmin=-np.pi, vmax=np.pi,
           extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]],
           cmap='twilight')

plt.axvline(r_max, color='white', linestyle='--', linewidth=1)
plt.axhline(tau_max, color='white', linestyle='--', linewidth=1)
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"Delay ($\mu$s)")
plt.colorbar(label="Phase")
plt.show()

plt.plot(delay, phase_sub.T[:,r_max_idx], ".", ms=2, c="k")
plt.xlabel(r"Delay ($\mu$s)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(delay[0], delay[-1])
plt.show()

plt.plot(fringe_rate, phase_sub.T[tau_max_idx, :], ".", ms=2, c="k")
plt.xlabel(r"Delay ($\mu$s)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(fringe_rate[0], fringe_rate[-1])
plt.show()

plt.plot(delay, power_sub[r_max_idx,:], ".", ms=2, c="k")
plt.xlabel(r"Delay ($\mu$s)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(delay[0], delay[-1])
plt.yscale("log")
plt.show()

plt.plot(fringe_rate, power_sub[:, tau_max_idx], ".", ms=2, c="k")
plt.xlabel("Fringe Rate (Hz)")
plt.ylabel(r"$|\hat{F}|$")
plt.xlim(fringe_rate[0], fringe_rate[-1])
plt.yscale("log")
plt.show()