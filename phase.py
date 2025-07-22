import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec

# Path to the MeasurementSet (MS)
data_MS  = "../Data/n24l2.ms"
#data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

# Open MS
data = table(data_MS)

n_rows  = int(len(data))
n_pol   = 0

time    = np.unique(data.getcol("TIME"))
t0      = time[0]
time   -= t0
nt      = len(time)

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
n_spw   = len(spectral_window)
freq    = spectral_window[0]['CHAN_FREQ']/1.e6
nf      = len(freq)
spectral_window.close()

antennas = table(data_MS+"/ANTENNA")
nameant  = antennas.getcol("NAME")
nant     = len(nameant)
antennas.close()

ant1    = 0
ant2    = 10
nSpW    = 1

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
    exit()

t1 = data.query(f'DATA_DESC_ID == {nSpW} AND ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')   # do row selection
data.close()

#vis_mod = t1.getcol('MODEL_DATA')[:, :, 0]   # (channel, pol [XX, XY, YX, YX])
vis = 0.5*(t1.getcol('DATA')[:, :, 0] + t1.getcol('DATA')[:, :, 3])   # (channel, pol [XX, XY, YX, YX])
flg = t1.getcol('FLAG')[:, :, 0]*t1.getcol('FLAG')[:, :, 3]   # (channel, pol [XX, XY, YX, YX])
ttt = t1.getcol('TIME')[:]  # (channel, pol [XX, XY, YX, YX])
uvw = t1.getcol('UVW')

vis[flg] = 0.0
uvw_mean = np.mean(np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2 + uvw[:, 2]**2))


#mod = t1.getcol('MODEL_DATA')[:, :, n_pol]   # (channel, pol [XX, XY, YX, YX])
#phase = np.angle(np.exp(1j*(np.angle(vis)-np.angle(mod))))
phase = np.angle(vis)#-np.angle(vis_mod)
# Choose indices to extract 1D slices

print(time.shape, freq.shape, phase.shape)
# Create figure and grid layout
fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Baseline {nameant[ant1]}-{nameant[ant2]}" + fr"   uv$[\lambda]$ = {uvw_mean:.0f}", fontsize=16, y=0.98)
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
              wspace=0.05, hspace=0.05)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.plot(time, np.mean(phase, axis=1), ".", color='purple')
ax_top.set_xlim(time[0], time[-1])
ax_top.set_ylabel("Phase")
ax_top.set_xticks([])

ax_right = fig.add_subplot(gs[1, 1])
ax_right.plot(np.mean(phase, axis=0), freq, ".", color='green')
ax_right.set_ylim(freq[0], freq[-1])
ax_right.set_xlabel("Phase")
ax_right.set_yticks([])

ax_main = fig.add_subplot(gs[1, 0])

im = ax_main.imshow(phase.T, aspect='auto', origin='lower', vmax=np.pi, vmin=-np.pi,
                    extent=[time[0], time[-1], freq[0], freq[-1]],
                    cmap='twilight')
im.cmap.set_bad(color='gray')  # Black for NaNs
ax_main.set_xlabel("Time (s)")
ax_main.set_ylabel("Frequency (MHz)")
cbar = fig.colorbar(im, ax=[ax_main, ax_top, ax_right], orientation='vertical',
                    pad=0.02, label="Phase (rads)")

plt.show()

fig = plt.figure(figsize=(12, 8))
im =plt.imshow(np.abs(vis).T, aspect='auto', origin='lower',
                    extent=[time[0], time[-1], freq[0], freq[-1]],
                    cmap='viridis')
im.cmap.set_bad(color='gray')  # Black for NaNs
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
cbar = fig.colorbar(im, orientation='vertical',
                    pad=0.02, label="Phase (rads)")
plt.show()