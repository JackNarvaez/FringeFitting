import datetime
import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import minimize
from GlobalFF import wrap_phase, SNR

def S3(vis_loc, wgt_loc, time, freq, params):
    phi0  = wrap_phase(params[0])
    r     = params[1]/tunit
    tau   = params[2]/funit
    Dt    = time - time[0]
    Df    = freq - freq[0]
    Eijk  = np.exp(1j*(phi0 + 2*np.pi*np.add.outer(r*Dt,tau*Df)))
    S2_t  = wgt_loc*np.abs(vis_loc-Eijk)**2
    S     = np.sum(S2_t)
    return S

def objective(params, vis_loc, wgt_loc, time, freq):
    return S3(vis_loc, wgt_loc, time, freq, params)

# Path to the MeasurementSet (MS)
data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
# data_MS  = "../Data/ILTJ123441.23+314159.4_141MHz_uv.dp3-concat"

# data_MS = "../Data/n24l2.ms"

data    = table(data_MS)
n_pol   = 0

time    = np.unique(data.getcol("TIME"))
t0      = time[0]
dt      = time[1]-time[0]
time   -= t0
nt      = len(time)

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freq    = spectral_window[0]['CHAN_FREQ']
nf      = len(freq)
spectral_window.close()

antennas = table(data_MS+"/ANTENNA")
nameant  = antennas.getcol("NAME")
nant     = len(nameant)
nbls     = (nant*(nant-1))//2
antennas.close()

nscale   = 8
ant1     = 10
ant2     = 26
lsm      = True
tunit    = 1e3
funit    = 1e9
nSpW     = 0

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant1}")
    exit()

t1  = data.query(f'DATA_DESC_ID == {nSpW} AND ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')   # do row selection
# print(t1.nrows(), data.nrows()/4/(nbls+nant))
data.close()

vis_global = 0.5*(t1.getcol('DATA')[:, :, 0] + t1.getcol('DATA')[:, :, 3])
# mod_global = 0.5*(t1.getcol('MODEL_DATA')[:, :, 0] + t1.getcol('MODEL_DATA')[:, :, 3])
flg_global = t1.getcol('FLAG')[:, :, 0]* t1.getcol('FLAG')[:, :, 3]

if "WEIGHT_SPECTRUM" in t1.colnames():
    wgt_global = 0.5*(t1.getcol('WEIGHT_SPECTRUM')[:, :, 0] + t1.getcol('WEIGHT_SPECTRUM')[:, :, 3])
else:
    # Use WEIGHT and broadcast it across channels
    weights = 0.5*(t1.getcol('WEIGHT')[:, 0] + t1.getcol('WEIGHT')[:, 3])
    wgt_global = weights[:, np.newaxis].repeat(nf, axis=1)

if lsm:
    wgt_global[flg_global] = 0.0
vis_global[flg_global] = 0.0
phase_global = np.angle(vis_global)
# model_global = np.angle(mod_global)

dim_nt, dimf = flg_global.shape
print("Flagged data:", np.sum(flg_global)/(dim_nt*dimf)*100)
ff1 = 0
ff2 = int(len(freq)/1)-1

intvals_t = 8
intvals_f = 1
fff = int(len(freq)/intvals_f)-1

DT = np.diff(time)
tindex = [0]
tindex.extend(np.where(DT > dt)[0] + 1)
tindex.append(nt)

Numdivs = len(tindex)-1

if (Numdivs > 1) and (Numdivs<3*intvals_t):
    intvals_t = Numdivs
    time_edgs = tindex
else:
    time_edgs = np.arange(0, nt, nt//intvals_t)
    time_edgs[-1] = nt

params    = np.zeros([intvals_t, intvals_f, 3])
freq_edgs = np.zeros(intvals_f)
vis_cal = np.zeros([nt, nf], dtype='c16')
print(vis_global[time_edgs[1]+10, 10])
plt.figure(figsize=(8, 6))
plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
plt.imshow(np.angle(vis_global).T, aspect='auto', origin='lower', cmap="twilight", vmin=-np.pi, vmax=np.pi,
        )
plt.xlabel("Time")
plt.ylabel(r"Frequency")
plt.colorbar(label="Phase")
plt.show()

for tint in range(intvals_t):
    tt1 = time_edgs[tint]
    tt2 = time_edgs[tint+1]
    print("\n", tint, f"\t tsteps: {tt2}-{tt1}")
    for fint in range(intvals_f):
        ff1 = fint*fff
        ff2 = ff1+fff

        if (fint==intvals_f-1):
            ff2 = nf-1
        if (tint == 0):
            freq_edgs[fint] = freq[ff2]
        vis = np.exp(1j*(phase_global[tt1:tt2, ff1:ff2+1]))# - model_global[tt1:tt2, ff1:ff2+1]))   # (channel, pol [RR, RL, LR, LL])
        print(vis.shape)
        if False:
            plt.figure(figsize=(8, 6))
            plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
            plt.imshow(np.angle(vis_global[tt1:tt2]).T, aspect='auto', origin='lower', cmap="twilight",
                        vmin=-np.pi, vmax=np.pi)
            plt.xlabel("Time")
            plt.ylabel(r"Frequency")
            plt.colorbar(label="Phase")
            plt.show()

        ####### FFT ######
        nt_sub, nf_sub = vis.shape

        # 2D FFT
        F_sub = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf_sub]))
        aa, bb = F_sub.shape
        F_sub[aa//2, bb//2] = 1e-15
        power_sub = np.abs(F_sub)
        phase_sub = np.angle(F_sub)


        print(SNR(power_sub))

        # ---------------------
        # FFT axes
        # ---------------------
        time_new = time[tt1:tt2]
        dt = (time_new[1] - time_new[0])             # time resolution [s]

        freq_new = freq[ff1:ff2]
        df = (freq_new[1] - freq_new[0])           # freq resolution [Hz]

        fringe_rate = fftshift(fftfreq(nscale*nt_sub, dt))  # in Hz
        delay       = fftshift(fftfreq(nscale*nf_sub, df))  # in [mu s]

        if False:
            plt.figure(figsize=(8, 6))
            plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
            plt.imshow(power_sub.T, aspect='auto', origin='lower', norm="log", vmax=4, vmin=1e-5,
                       extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]],
                       cmap='plasma')
            plt.xlabel("Fringe Rate (Hz)")
            plt.ylabel(r"Delay ($\mu$s)")
            plt.colorbar(label="Amplitude")
            plt.show()
        # ---------------------
        # Find argmax
        # ---------------------
        r_max_idx, tau_max_idx = np.unravel_index(np.argmax(power_sub), power_sub.shape)

        print("\t", r_max_idx, aa//2, tau_max_idx, bb//2)

        params[tint, fint, 0] = phase_sub[r_max_idx, tau_max_idx]
        params[tint, fint, 1] = fringe_rate[r_max_idx]*tunit
        params[tint, fint, 2] = delay[tau_max_idx]*funit

        print(f"phase: {params[tint, fint, 0]:.3f}\t rate: {params[tint, fint, 1]:.3f}\t delay: {params[tint, fint, 2]:.3f}\n")

        INI_TIME = datetime.datetime.now()
        if lsm:
            wgt = wgt_global[tt1:tt2+1, ff1:ff2+1]   # (channel, pol [RR, RL, LR, LL])
            params0 = params[tint, fint].copy()
            S0      = S3(vis, wgt, time_new, freq_new,params0)
            result  = minimize(objective, params[tint, fint], args=(vis, wgt, time_new, freq_new), method='L-BFGS-B', options={'maxiter':100})
            print(params[tint, fint], result.x)
            params[tint, fint] = result.x
            S1 = result.fun
            print(f"{params0[0]:.5f}\t\t{params0[1]:.5f}\t\t{params0[2]:.5f}")
            print(f"{params[tint, fint, 0]:.5f}\t\t{params[tint, fint, 1]:.5f}\t\t{params[tint, fint, 2]:.5f}")
            print(f"res: {100*(1.-S1/S0):.5f} %")
        
        print("TIME: ", (datetime.datetime.now()-INI_TIME).seconds)

        #### Calibrate phase
        G_ff_inv = np.exp(-1j*(params[tint, fint, 0]+2*np.pi*(params[tint, fint, 1]/tunit*(time[tt1:tt2, np.newaxis]-time[tt1]) + params[tint, fint, 2]/funit*(freq[ff1:ff2+1]-freq[ff1]))))
        vis_cal[tt1:tt2, ff1:ff2+1] = vis*G_ff_inv

def plot_phase(vis_global, time, freq, wgt=None, BL="01", showw=False):
    """
    Plot phase heatmap and statistics (mean, std) along time and frequency axes.

    Parameters:
        vis_global (ndarray): Complex visibility array.
        time (ndarray): Time axis.
        freq (ndarray): Frequency axis.
        showw (bool): If True, display the figure with plt.show().
    """
    phase_global = np.angle(vis_global)

    if (wgt is not None):
         weights = wgt.flatten()
    else:
        weights = None
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
    temp_time   = np.linspace(time[0], time[-1], len(time))
    time_rep    = np.repeat(temp_time[:, np.newaxis], phase_global.shape[1], axis=1).flatten()
    nbins_time  = len(time)
    nbins_phase = 100

    ax_top2.hist2d(
        time_rep, phase_global.flatten(),
        weights = weights,
        bins=[nbins_time, nbins_phase],
        range=[None, [-np.pi, np.pi]],
        cmap='viridis',cmin=0.1,
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
        weights = weights,
        bins=[nbins_phase, len(freq)],
        range=[[-np.pi, np.pi], [freq[0], freq[-1]]],
        cmap='viridis', cmin=0.1,
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

vis_cal[flg_global] = 0.0
# final_vis = np.exp(1j*(phase_global-model_global))
plot_phase(vis_global, time, freq, wgt=wgt_global, BL=f"{nameant[ant1]}-{nameant[ant2]} VIS", showw=False)
# plot_phase(mod_global, time, freq, wgt=wgt_global, BL=f"{nameant[ant1]}-{nameant[ant2]} MOD", showw=False)
# plot_phase(final_vis, time, freq, wgt=wgt_global, BL=f"{nameant[ant1]}-{nameant[ant2]} VIS-MOD", showw=False)
plot_phase(vis_cal, time, freq, wgt=wgt_global, BL=f"{nameant[ant1]}-{nameant[ant2]} CAL", showw=True)