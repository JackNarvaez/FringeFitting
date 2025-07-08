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
refant   = 0
ant1     = 10
ant2     = 60

if (refant >= ant1):
    print(f"Error: Ant1: {refant} must be smaller than Ant1: {ant1}")
    exit()
if (refant >= ant2):
    print(f"Error: Ant1: {refant} must be smaller than Ant2: {ant2}")
    exit()

t1  = data.query('ANTENNA1 == '+str(refant) +' AND ANTENNA2 == '+str(ant1))   # do row selection
t2  = data.query('ANTENNA1 == '+str(refant) +' AND ANTENNA2 == '+str(ant2))   # do row selection

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

        ##### ANT1 #####
        vis_dat1 = t1.getcol('DATA')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
        flg1 = t1.getcol('FLAG')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
        vis1 = np.exp(1j*np.angle(vis_dat1))
        vis1[flg1] = 0.0

        ##### ANT2 #####
        vis_dat2 = t2.getcol('DATA')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
        flg2 = t2.getcol('FLAG')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
        vis2 = np.exp(1j*np.angle(vis_dat2))
        vis2[flg2] = 0.0

        ####### FFT ######
        nt1, nf1 = vis1.shape
        F1_sub  = fftshift(fft2(vis1, norm='ortho', s=[nscale*nt1, nscale*nf1]))
        power_1 = np.abs(F1_sub)
        phase_1 = np.angle(F1_sub)

        F2_sub  = fftshift(fft2(vis2, norm='ortho', s=[nscale*nt1, nscale*nf1]))
        power_2 = np.abs(F2_sub)
        phase_2 = np.angle(F2_sub)
        phase   = np.angle(np.exp(1j*(np.angle(vis2) - np.angle(vis1))))

        # ---------------------
        # FFT axes
        # ---------------------
        time_new = np.linspace(time[tt1], time[tt2], nt1)
        dt = (time_new[1] - time_new[0])             # time resolution [s]

        freq_new = np.linspace(freq[ff1], freq[ff2], nf1)
        df = (freq_new[1] - freq_new[0])           # freq resolution [Hz]
        
        fringe_rate = fftshift(fftfreq(nscale*nt1, dt))  # in Hz
        delay       = fftshift(fftfreq(nscale*nf1, df))*1e6  # in [mu s]

        # ---------------------
        # Find argmax
        # ---------------------
        peak_1 = np.unravel_index(np.argmax(power_1), power_1.shape)
        r_max_1, tau_max_1 = peak_1

        phase_max_1 = phase_1[r_max_1, tau_max_1]
        r_max_1 = fringe_rate[r_max_1]   # Hz
        tau_max_1 = delay[tau_max_1]     # seconds

        peak_2 = np.unravel_index(np.argmax(power_2), power_2.shape)
        r_max_2, tau_max_2 = peak_2

        phase_max_2 = phase_2[r_max_2, tau_max_2]
        r_max_2 = fringe_rate[r_max_2]   # Hz
        tau_max_2 = delay[tau_max_2]     # seconds

        phase_max[pp, nnn] = np.angle(np.exp(1j*(phase_max_2 -phase_max_1)))
        delay_max[pp, nnn] = (tau_max_2 - tau_max_1)*1e3
        rate_max[pp, nnn]  = (r_max_2 - r_max_1)*1e3
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

f_idx =int(nf1/2)
t_idx = int(nt1/3)

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