import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
plt.style.use('dark_background')


rcParams.update({
    "font.size": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "lines.markersize": 7,
    "lines.linewidth": 4,
    "figure.figsize": (14, 8)
})

# Path to the MeasurementSet (MS)
# data_MS  = "../Data/n24l2.ms"
data_MS = "../Data/ILTJ123441.23+314159.4_141MHz_uv.dp3-concat"
# data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

# Open MS
data = table(data_MS)

n_rows  = int(len(data))

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
antpos   = antennas.getcol("POSITION")
print(antennas.getcoldesc("POSITION")['keywords']['QuantumUnits'][0])
nant     = len(nameant)
antennas.close()

ant1    = 2
ant2    = 25
nSpW    = 0

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
    exit()

t1 = data.query(f'DATA_DESC_ID == {nSpW} AND ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')   # do row selection
data.close()

vis = 0.5*(t1.getcol('DATA')[:, :, 0] + t1.getcol('DATA')[:, :, 3])   # (channel, pol [XX, XY, YX, YX])
vis_mod = 0.5*(t1.getcol('MODEL_DATA')[:, :, 0] + t1.getcol('MODEL_DATA')[:, :, 3])   # (channel, pol [XX, XY, YX, YX])
flg = t1.getcol('FLAG')[:, :, 0]*t1.getcol('FLAG')[:, :, 3]   # (channel, pol [XX, XY, YX, YX])
ttt = t1.getcol('TIME')[:]  # (channel, pol [XX, XY, YX, YX])

phase = np.angle(vis)-np.angle(vis_mod)#-np.angle(vis_mod)

fff = 200
ttt = 80

fig, ax = plt.subplots()
im = ax.imshow(phase.T, cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi, aspect="auto", 
          extent=[time[0], time[-1], freq[0], freq[-1]])
plt.colorbar(im, ax=ax)
ax.axvline(time[fff], ls="--", lw=1)
ax.axhline(freq[ttt], ls="--", lw=1)
ax.set_ylim(freq[0], freq[-1])
ax.set_xlim(time[0], time[-1])

ax.set_ylabel("Frequency [MHz]")
ax.set_xlabel("Time [H]")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
ax.plot(freq, phase[fff], ".", color='deepskyblue', ms=10)

ax.set_ylim(-np.pi, np.pi)
ax.set_xlim(freq[0], freq[-1])

ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Phase [rad]")
ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
# plt.savefig("FF_theory.png", dpi=300, transparent=True)
plt.show()


fig, ax = plt.subplots()
ax.plot(time, phase[:, ttt], ".", color='deepskyblue', ms=10)

ax.set_ylim(-np.pi, np.pi)
ax.set_xlim(time[0], time[-1])

ax.set_xlabel("Time [s]")
ax.set_ylabel("Phase [rad]")
ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
# plt.savefig("FF_theory.png", dpi=300, transparent=True)
plt.show()

