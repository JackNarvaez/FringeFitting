"""
Fringe Fitting and Calibration for Artificial Visibility Data
==============================================================

This program performs phase calibration of artificial visibility data
using fringe fitting techniques to calculate phase, rate, and delay 
based on Fast Fourier Transforms and Least Squares Minimization.
"""

import time as tm
import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import minimize

def SNR(dataFFT):
    """
    Compute the Signal-to-Noise Ratio (SNR).

    Parameters:
        dataFFT (ndarray): 2D array representing the FFT power spectrum.

    Returns:
        float: Ratio of the peak value to the mean, i.e., max / mean.
    """

    return np.max(dataFFT)/np.mean(dataFFT)

def snr_aips(peak, xcount_):
    if (peak > 0.999*xcount_):
        peak = 0.999*xcount_
        print("peak > 0.999*sumw_")

    x = np.pi/2 *peak/xcount_
    # The magic numbers in the following formula are from AIPS FRING
    cwt = (np.tan(x)**1.163) * np.sqrt(xcount_)
    return cwt

def FFT_guess(vis, nscale, nt_sub, nf):
    """
    Estimate initial fringe parameters (phase, rate, delay) using 2D FFT.

    Parameters:
        vis (ndarray): Subset of visibility data (time x frequency).
        nscale (int): Zero-padding scale factor for FFT.
        nt_sub (int): Number of time samples in the sub-block.
        nf (int): Number of frequency channels.

    Returns:
        tuple: (phi0, r_idx, tau_idx, snr) where:
            phi0 (float): Estimated phase offset.
            r_idx (int): Index of fringe rate peak.
            tau_idx (int): Index of delay peak.
            snr (float): Signal-to-noise ratio of the FFT.
    """

    F_sub  = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf]))
    power_sub = np.abs(F_sub)

    r_max_idx, tau_max_idx = np.unravel_index(np.argmax(power_sub), power_sub.shape)
    phi0 = np.angle(F_sub[r_max_idx, tau_max_idx])
    #snr_ = SNR(power_sub)
    shp = power_sub.shape
    xcount_ = shp[0]*shp[1]
    snr_ = snr_aips(np.abs(F_sub[r_max_idx, tau_max_idx]), xcount_)
    return phi0, r_max_idx, tau_max_idx, snr_

def wrap_phase(phase):
        """Fast phase wrapping"""
        return ((phase + np.pi) % (2 * np.pi)) - np.pi

def S3(vis_loc, time, freq, prms):
    """
    Compute the sum-of-squares cost function between model and measured visibilities.

    Parameters:
        vis_loc (ndarray): Observed visibility data for a baseline.
        time (ndarray): Time array (1D).
        freq (ndarray): Frequency array (1D).
        prms0 (array): Parameters (phi0, r, tau) of antenna 0.
        prms1 (array): Parameters of antenna 1.

    Returns:
        float: Sum of squared phase errors.
    """
    
    phi0  = wrap_phase(prms[0])
    r     = prms[1]/tunit
    tau   = prms[2]/funit
    
    Dt    = time - time[0]
    Df    = freq - freq[0]
    Eijk  = np.exp(1j*(phi0 + 2*np.pi*np.add.outer(r*Dt,tau*Df)))
    S2_t  = np.abs(vis_loc-Eijk)**2
    S     = np.sum(S2_t)
    return S

def objective(prms, vis_01, vis_02, vis_12, time, freq):
    """
    Cost function for least-squares optimization of fringe parameters across all baselines.

    Parameters:
        prms (ndarray): Flattened array of parameters [phi0, r, tau] for all antennas.
        vis_01, vis_02, vis_12 (ndarray): Visibilities for the three baselines.
        time, freq (ndarray): Time and frequency axes.

    Returns:
        float: Sum of squared residuals across all three baselines.
    """
    prms0 = prms[3:6]-prms[0:3]
    prms1 = prms[6:9]-prms[0:3]
    prms2 = prms[6:9]-prms[3:6]
    S_01 = S3(vis_01, time, freq, prms0)
    S_02 = S3(vis_02, time, freq, prms1)
    S_12 = S3(vis_12, time, freq, prms2)
    return S_01+S_02+S_12

def calibrate(vis_global, vis_cal, prms0, prms1, time, freq, intvals_t, ttt, model=False):
    """
    Apply phase correction to global visibilities.

    Parameters:
        vis_global (ndarray): Raw visibility data (complex).
        vis_cal (ndarray): Output array for calibrated visibility.
        prms0 (ndarray): Parameters of antenna 0 for each time block.
        prms1 (ndarray): Parameters of antenna 1 for each time block.
        time (ndarray): Time axis (1D).
        freq (ndarray): Frequency axis (1D).
        intvals_t (int): Number of time intervals to divide the data into.
        ttt (int): Time block size (in number of samples).
    """

    prms = prms1 - prms0
    vis_model = np.zeros_like(vis_global)
    for tint in range(intvals_t):
        tt1 = tint*ttt
        tt2 = tt1+ttt
        if (tint==intvals_t-1):
            tt2 = nt-1
        nt_sub = tt2+1-tt1
        G_ff_inv = np.exp(-1j*(prms[tint, 0]+2*np.pi*(prms[tint, 1]/tunit*(time[tt1:tt2+1, np.newaxis]-time[tt1]) + prms[tint, 2]/funit*(freq-freq[0]))))
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
time    = np.unique(data.getcol("TIME"))
t0      = time[0]
time   -= t0
nt      = len(time)
data.close()

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freq    = spectral_window[0]['CHAN_FREQ']
nf      = len(freq)
spectral_window.close()

########## PARAMETERS ##########
nscale  = 4
nant    = 3

#3 Baselines: 01 - 12 - 02
r       = np.array([1e-3, 1.5e-3, 0.0])  # 1e-3 = Hz 1 ms
tau     = np.array([1e-7, 3e-7, 0.0])    # 1e-9 = 1 ns
phi0    = np.array([0.0, -np.pi/3, 0.0])
r[-1]   = r[0] + r[1]
tau[-1] = tau[0] + tau[1]
phi0[-1]= np.angle(np.exp(1j*(phi0[0] + phi0[1])))

noise   = 1
lsm     = True
tunit   = 1e3
funit   = 1e9
intvals_t = 8
params  = np.zeros([nant, intvals_t, 3])
snr_bef = np.zeros((intvals_t, nant))
snr_aft = np.zeros((intvals_t, nant))
ttt     = nt//intvals_t - 1

dt = time[1] - time[0]           # time resolution [s]
df = freq[1] - freq[0]           # freq resolution [Hz]

delay   = fftshift(fftfreq(nscale*nf, df))  # in [mu s]

########## VISIBILITIES ##########
vis_global_01 = np.exp(1j*(phi0[0] + 2*np.pi*(r[0]*time[:, np.newaxis] + tau[0]*(freq-freq[0])) + noise*np.random.normal(0,1,(nt, nf))))
vis_global_02 = np.exp(1j*(phi0[2] + 2*np.pi*(r[2]*time[:, np.newaxis] + tau[2]*(freq-freq[0])) + noise*np.random.normal(0,1,(nt, nf))))
vis_global_12 = np.exp(1j*(phi0[1] + 2*np.pi*(r[1]*time[:, np.newaxis] + tau[1]*(freq-freq[0])) + noise*np.random.normal(0,1,(nt, nf))))
vis_cal_01 = np.zeros_like(vis_global_01)
vis_cal_02 = np.zeros_like(vis_global_02)
vis_cal_12 = np.zeros_like(vis_global_12)

##########  Initial Guess (FFT) ##########
for tint in range(intvals_t):
    tt1 = tint*ttt
    tt2 = tt1+ttt
    if (tint==intvals_t-1):
        tt2 = nt-1
    nt_sub = tt2+1-tt1

    time_new    = np.linspace(time[tt1], time[tt2], nt_sub)
    fringe_rate = fftshift(fftfreq(nscale*nt_sub, dt))  # in Hz

    vis    = vis_global_01[tt1:tt2+1]
    phi0, r_max_idx, tau_max_idx, snr_ = FFT_guess(vis, nscale, nt_sub, nf)
    params[1, tint, 0] = phi0
    params[1, tint, 1] = fringe_rate[r_max_idx]*tunit
    params[1, tint, 2] = delay[tau_max_idx]*funit
    snr_bef[tint, 0]   = snr_

    vis    = vis_global_02[tt1:tt2+1]
    phi0, r_max_idx, tau_max_idx, snr_ = FFT_guess(vis, nscale, nt_sub, nf)
    params[2, tint, 0] = phi0
    params[2, tint, 1] = fringe_rate[r_max_idx]*tunit
    params[2, tint, 2] = delay[tau_max_idx]*funit
    snr_bef[tint, 1]   = snr_

    vis    = vis_global_12[tt1:tt2+1]
    F_sub  = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf]))
    pwr_l  = np.abs(F_sub)
    snr_bef[tint, 2]   = SNR(pwr_l)

########## Least-Squares Minimization ##########
INI_TIME = tm.time()
if lsm:
    lsm_imp = np.zeros(intvals_t)
    for tint in range(intvals_t):
        tt1 = tint*ttt
        tt2 = tt1+ttt
        if (tint==intvals_t-1):
            tt2 = nt-1
        nt_sub = tt2+1-tt1
        time_new    = np.linspace(time[tt1], time[tt2], nt_sub)
        x0 = params[:, tint].flatten()
        S0      = objective(x0, vis_global_01[tt1:tt2+1], vis_global_02[tt1:tt2+1], vis_global_12[tt1:tt2+1], time_new, freq)
        bounds = []
        for i in range(nant):
            bounds.extend([(-np.pi, np.pi), (None, None), (None, None)])  # phi0, r, tau
            
        result  = minimize(objective, x0, args=(vis_global_01[tt1:tt2+1], vis_global_02[tt1:tt2+1], vis_global_12[tt1:tt2+1], time_new, freq), method='L-BFGS-B', bounds=bounds, options={'maxiter':500})
        params[:, tint] = result.x.reshape(3, 3)
        ST = result.fun
        lsm_imp[tint] = 100*(1.-ST/S0)

print("TIME: ", tm.time()-INI_TIME)

######### Calibrate phase #########
model_01 = calibrate(vis_global_01, vis_cal_01, params[0], params[1], time, freq, intvals_t, ttt, model=True)
model_02 = calibrate(vis_global_02, vis_cal_02, params[0], params[2], time, freq, intvals_t, ttt, model=True)
model_12 = calibrate(vis_global_12, vis_cal_12, params[1], params[2], time, freq, intvals_t, ttt, model=True)


##########  Final SNR ##########
for tint in range(intvals_t):
    tt1 = tint*ttt
    tt2 = tt1+ttt
    if (tint==intvals_t-1):
        tt2 = nt-1
    nt_sub = tt2+1-tt1

    vis    = vis_cal_01[tt1:tt2+1]
    F_sub  = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf]))
    pwr_l  = np.abs(F_sub)
    snr_aft[tint, 0]   = SNR(pwr_l)

    vis    = vis_cal_02[tt1:tt2+1]
    F_sub  = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf]))
    pwr_l  = np.abs(F_sub)
    snr_aft[tint, 1]   = SNR(pwr_l)

    vis    = vis_cal_12[tt1:tt2+1]
    F_sub  = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf]))
    pwr_l  = np.abs(F_sub)
    snr_aft[tint, 2]   = SNR(pwr_l)

#### Plot Results

for tint in range(intvals_t):
    print(f"Interval: {tint}\t Imp. LSM: {lsm_imp[tint]:.5f} % \t Init. SNR: {np.mean(snr_bef[tint]):.3f}\tFinal SNR: {np.mean(snr_aft[tint]):.3f}")

plot_phase(vis_global_01, time, freq, BL="01", showw=False)
plot_phase(vis_global_02, time, freq, BL="02", showw=False)
plot_phase(vis_global_01, time, freq, BL="12", showw=False)

plot_phase(model_01, time, freq, BL="01", showw=False)
plot_phase(model_02, time, freq, BL="02", showw=False)
plot_phase(model_01, time, freq, BL="12", showw=True)

plot_phase(vis_cal_01, time, freq, BL="01", showw=False)
plot_phase(vis_cal_02, time, freq, BL="02", showw=False)
plot_phase(vis_cal_01, time, freq, BL="12", showw=True)