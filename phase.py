import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec

# Path to the MeasurementSet (MS)
data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

# Open MS
data = table(data_MS)

n_rows  = int(len(data))
n_pol   = 0

time    = np.unique(data.getcol("TIME"))
t0      = time[0]
time   -= t0
nt      = len(time)

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freq    = spectral_window[0]['CHAN_FREQ']/1.e6
nf      = len(freq)
spectral_window.close()

antennas = table(data_MS+"/ANTENNA")
nant     = len(antennas.getcol("NAME"))
antennas.close()

ant1    = 0
ant2    = 43

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
    exit()

t1 = data.query('ANTENNA1 == '+str(ant1) +' AND ANTENNA2 == '+str(ant2))   # do row selection
data.close()

vis = t1.getcol('CALIBRATED_DATA')[:, :, n_pol]   # (channel, pol [XX, XY, YX, YX])
flg = t1.getcol('FLAG')[:, :, n_pol]   # (channel, pol [XX, XY, YX, YX])
uvw = t1.getcol('UVW')

uvw_mean = np.mean(np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2 + uvw[:, 2]**2))

#mod = t1.getcol('MODEL_DATA')[:, :, n_pol]   # (channel, pol [XX, XY, YX, YX])
#phase = np.angle(np.exp(1j*(np.angle(vis)-np.angle(mod))))
phase = np.angle(vis)
# Choose indices to extract 1D slices
f_idx = int(nf/2)  # middle frequency
t_idx = int(nt/2)  # middle time

# Create figure and grid layout
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Baseline "+str(ant1)+"-"+str(ant2) + fr"   uv$[\lambda]$ = {uvw_mean:.0f}", fontsize=16, y=0.98)
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
              wspace=0.05, hspace=0.05)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.plot(time, phase[:, f_idx], ".", color='purple')
ax_top.set_ylabel("Phase")
ax_top.set_xticks([])
ax_top.set_title(f"@ {freq[f_idx]:.1f} MHz")

ax_right = fig.add_subplot(gs[1, 1])
ax_right.plot(phase[t_idx, :], freq, ".", color='green')
ax_right.set_xlabel("Phase")
ax_right.set_yticks([])
ax_right.set_title(f"@ {time[t_idx]:.1f} s")

ax_main = fig.add_subplot(gs[1, 0])

masked_phase = np.ma.masked_array(phase, mask=flg)  # Match shape
im = ax_main.imshow(masked_phase.T, aspect='auto', origin='lower',
                    extent=[time[0], time[-1], freq[0], freq[-1]],
                    cmap='seismic')
im.cmap.set_bad(color='gray')  # Black for NaNs
ax_main.axhline(freq[f_idx], ls="--",color="k")
ax_main.axvline(time[t_idx], ls="--", color="k")
ax_main.set_xlabel("Time (s)")
ax_main.set_ylabel("Frequency (MHz)")
cbar = fig.colorbar(im, ax=[ax_main, ax_top, ax_right], orientation='vertical',
                    pad=0.02, label="Phase (rads)")

plt.show()