import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq
from matplotlib import rcParams
plt.style.use('dark_background')


rcParams.update({
    "font.size": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "lines.markersize": 7,
    "lines.linewidth": 4,
    "figure.figsize": (8, 8)
})

def wrap_phase(phase):
        """Fast phase wrapping"""
        return ((phase + np.pi) % (2 * np.pi)) - np.pi

def calibrate(vis_global, vis_cal, prms0, time, freq, intvals_t, ttt, model=False):
    prms = prms0
    vis_model = np.zeros_like(vis_global)
    for tint in range(intvals_t):
        tt1 = tint*ttt
        tt2 = tt1+ttt
        if (tint==intvals_t-1):
            tt2 = nt-1
        G_ff_inv = np.exp(-1j*(prms[0]+2*np.pi*(prms[1]/tunit*(time[tt1:tt2+1, np.newaxis]-time[tt1]) + prms[2]/funit*(freq-freq[0]))))
        vis_cal[tt1:tt2+1] = vis_global[tt1:tt2+1]*G_ff_inv 
        vis_model[tt1:tt2+1] = G_ff_inv
    if model:
        return vis_model


def plot_phase(vis_global, time, freq, BL="01", showw=False):
    """
    Plot phase heatmap and statistics (mean, std) along time and frequency axes.

    Parameters:
        vis_global (ndarray): Complex visibility array.
        time (ndarray): Time axis.
        freq (ndarray): Frequency axis.
        showw (bool): If True, display the figure with plt.show().
    """

    phase_global = np.angle(vis_global)

    # Create figure and grid layout
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Baseline: {BL}")
    gs  = GridSpec(3, 3, width_ratios=[8, 1, 1], height_ratios=[1, 1, 4],
                wspace=0.05, hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(time, np.std(phase_global, axis=1), ".", color='purple')
    ax_top.set_xlim(time[0], time[-1])
    ax_top.set_ylabel("STD")
    ax_top.set_xticks([])

    ax_top2     = fig.add_subplot(gs[1, 0])
    time_rep    = np.repeat(time[:, np.newaxis], phase_global.shape[1], axis=1).flatten()
    nbins_phase = 100

    ax_top2.hist2d(
        time_rep, phase_global.flatten(),
        bins=[nt//4, nbins_phase],
        range=[[time[0], time[-1]], [-np.pi, np.pi]],
        cmap='viridis', cmin=1
    )

    ax_top2.set_ylabel("Phase")
    ax_top2.set_xlim(time[0], time[-1])
    ax_top2.set_ylim(-np.pi, np.pi)
    ax_top2.set_xticks([])

    ax_right = fig.add_subplot(gs[2, 2])
    ax_right.plot(np.std(phase_global,axis=0), freq, ".", color='green')
    ax_right.set_ylim(freq[0], freq[-1])
    ax_right.set_xlabel("STD")
    ax_right.set_yticks([])

    ax_right2 = fig.add_subplot(gs[2, 1])
    freq_rep    = np.repeat(freq[np.newaxis, :], phase_global.shape[0], axis=0).flatten()

    ax_right2.hist2d(
        phase_global.flatten(), freq_rep,
        bins=[nbins_phase, nf],
        range=[[-np.pi, np.pi], [freq[0], freq[-1]]],
        cmap='viridis', cmin=1
    )

    ax_right2.set_xlabel("Phase")
    ax_right2.set_ylim(freq[0], freq[-1])
    ax_right2.set_xlim(-np.pi, np.pi)
    ax_right2.set_yticks([])

    ax_main = fig.add_subplot(gs[2, 0])

    ax_main.imshow(phase_global.T, aspect='auto', origin='lower', vmax=np.pi, vmin=-np.pi,
                    extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    if showw:
        plt.show()

########## LOAD DATA ##########
data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data    = table(data_MS)
timw    = np.unique(data.getcol("TIME"))
time    = timw[:len(timw)]
t0      = time[0]
time   -= t0
nt      = len(time)
data.close()

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freq    = spectral_window[0]['CHAN_FREQ']
nf      = len(freq)
spectral_window.close()

########## PARAMETERS ##########
nscale  = 8

#3 Baselines: 01 - 12 - 02
r       = 3e-4  # 1e-3 = Hz 1 ms
tau     = 1e-7    # 1e-9 = 1 ns
phi0    = 0.0

noise   = 3
lsm     = True
tunit   = 1e3
funit   = 1e9
intvals_t = 1
params  = np.zeros(3)
ttt     = nt//intvals_t - 1

dt = time[1] - time[0]           # time resolution [s]
df = freq[1] - freq[0]           # freq resolution [Hz]

delay   = fftshift(fftfreq(nscale*nf, df))  # in [mu s]

########## VISIBILITIES ##########
vis = np.exp(1j*(phi0 + 2*np.pi*(r*time[:, np.newaxis] + tau*(freq-freq[0])) + noise*np.random.normal(0,1,(nt, nf))))
vis_cal = np.zeros_like(vis)

##########  Initial Guess (FFT) ##########

fringe_rate = fftshift(fftfreq(nscale*nt, dt))  # in Hz

F_sub  = fftshift(fft2(vis, s=[nscale*nt, nscale*nf]))
power_sub = np.abs(F_sub)
dimm = power_sub.shape

power_sub[dimm[0]//2, :] *= 0.0
power_sub[:, dimm[1]//2] *= 0.0

r_max_idx, tau_max_idx = np.unravel_index(np.argmax(power_sub), dimm)
peak = np.pi/2 * np.abs(F_sub[r_max_idx, tau_max_idx])/(nt*nf)
phi0 = np.angle(F_sub[r_max_idx, tau_max_idx])
xcount_ = nf*nt
print(peak, np.tan(peak)**1.163 * np.sqrt(xcount_/np.sqrt(xcount_/(nf*nt))))
params[0] = phi0
params[1] = fringe_rate[r_max_idx]*tunit
params[2] = delay[tau_max_idx]*funit

G_ff_inv = np.exp(-1j*(params[0]+2*np.pi*(params[1]/tunit*(time[:, np.newaxis]-time[0]) + params[2]/funit*(freq-freq[0]))))
vis_cal  = vis*G_ff_inv

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(np.angle(vis).T, aspect='auto', origin='lower', vmax=np.pi, vmin=-np.pi,
                    extent=[time[0]/3600, time[-1]/3600, freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')

cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Phase (rad)")

ax.set_ylim(freq[0]/1e6, freq[-1]/1e6)
ax.set_xlim(time[0]/3600, time[-1]/3600)

ax.set_xlabel("Time (H)")
ax.set_ylabel("Frequency (MHz)")
plt.tight_layout()
plt.savefig("Ftnu.png", dpi=500, transparent=True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(np.angle(G_ff_inv).T, aspect='auto', origin='lower', vmax=np.pi, vmin=-np.pi,
                    extent=[time[0]/3600, time[-1]/3600, freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')

cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Phase (rad)")

ax.set_ylim(freq[0]/1e6, freq[-1]/1e6)
ax.set_xlim(time[0]/3600, time[-1]/3600)

ax.set_xlabel("Time (H)")
ax.set_ylabel("Frequency (MHz)")
plt.tight_layout()
# plt.savefig("Model.png", dpi=500, transparent=True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(np.angle(vis_cal).T, aspect='auto', origin='lower', vmax=np.pi, vmin=-np.pi,
                    extent=[time[0]/3600, time[-1]/3600, freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')

cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Phase (rad)")

ax.set_ylim(freq[0]/1e6, freq[-1]/1e6)
ax.set_xlim(time[0]/3600, time[-1]/3600)

ax.set_xlabel("Time (H)")
ax.set_ylabel("Frequency (MHz)")
plt.tight_layout()
# plt.savefig("Calibrated.png", dpi=500, transparent=True)
plt.show()

mmax = np.max(power_sub)

fig, ax = plt.subplots(figsize=(10.5, 8))
img = ax.imshow(power_sub.T, aspect='auto', origin='lower', norm="log", vmax=mmax, vmin=mmax/10000,
                    extent=[fringe_rate[0]*tunit, fringe_rate[-1]*tunit, delay[0]*funit, delay[-1]*funit],
                    cmap='gist_heat')

cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Intensity")

ax.set_xlim(-8, 8)
ax.set_ylim(-800, 800)
ax.set_yticks([-500, 0, 500])

ax.set_xlabel("Rate (mHz)")
ax.set_ylabel("Delay (ns)")
plt.tight_layout()
# plt.savefig("FFT.png", dpi=300, transparent=True)

plt.show()