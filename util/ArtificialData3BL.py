"""
Fringe Fitting Calibration for Artificial Data
==============================================================

This program performs phase calibration of artificial visibility data
using fringe fitting techniques to calculate phase, rate, and delay 
based on Fast Fourier Transforms and Least Squares Minimization.

Number of baselines: 3
"""

import time as tm
import numpy as np
import matplotlib.pyplot as plt
from casacore.tables    import table
from numpy.fft          import fftshift, fft2, fftfreq
from scipy.optimize     import minimize, least_squares, differential_evolution
from ..GlobalFF         import snr_aips, wrap_phase, plot_phase, M_PI2, FLT_EPSILON
from scipy.optimize     import curve_fit

def gaussian(x, A, mu, sigma, C):
    """Gaussian + constant background"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

def FFT_guess(vis, nscale, nt_sub, nf):
    F_sub  = fftshift(fft2(vis, s=[nscale*nt_sub, nscale*nf]))
    power_sub = np.abs(F_sub)
    dimm = power_sub.shape

    power_sub[dimm[0]//2, :] = 0.0
    power_sub[:, dimm[1]//2] = 0.0

    r_max_idx, tau_max_idx = np.unravel_index(np.argmax(power_sub), dimm)
    phi0 = np.angle(F_sub[r_max_idx, tau_max_idx])

    snr_ = snr_aips(np.abs(F_sub[r_max_idx, tau_max_idx]), nt_sub*nf, nt_sub*nf, nt_sub*nf)



    slice_tau = np.abs(F_sub[r_max_idx, :])   # magnitude along tau
    x = np.arange(dimm[1])

    # initial guesses for fit
    A0 = slice_tau.max() - slice_tau.min()
    mu0 = tau_max_idx
    sigma0 = 5.0  # some reasonable width guess
    C0 = slice_tau.min()

    # fit Gaussian
    popt, _ = curve_fit(gaussian, x, slice_tau, p0=[A0, mu0, sigma0, C0])

    A_fit, mu_fit, sigma_fit, C_fit = popt
    # plt.figure()
    # plt.plot(x, slice_tau, ".", c="w", ms=1)
    # plt.plot(x, gaussian(x, A0, mu0, sigma0, C0), "-", c="g", lw=1)
    # plt.plot(x, gaussian(x, A_fit, mu_fit, sigma_fit, C_fit), "-", c="r", lw=1)
    # plt.yscale("log")
    # plt.ylabel("Intensity")
    # plt.xlabel("Delay")
    # plt.show()

    return phi0, r_max_idx, tau_max_idx, snr_, sigma_fit

def S3(vis_loc, time, freq, prms):
    phi0  = wrap_phase(prms[0])
    r     = prms[1]/tunit
    tau   = prms[2]/funit
    kdisp = prms[3]*kunit
    
    Dt    = time - time[0]
    Df    = freq - freq[0]
    kdisp = (prms[3]*kunit)*(1/freq + (freq - fmin - fmax)/(fmin*fmax))
    fCorr = tau*Df + kdisp
    
    Eijk  = np.exp(1j*(phi0 + 2*np.pi*np.add.outer(r*Dt,fCorr)))
    S2_t  = np.abs(vis_loc-Eijk)**2
    S     = np.sum(S2_t)
    return S

def objective(prms, vis_01, vis_02, vis_12, time, freq):
    prms0 = prms[4:8]-prms[0:4]
    prms1 = prms[8:12]-prms[0:4]
    prms2 = prms[8:12]-prms[4:8]
    S_01 = S3(vis_01, time, freq, prms0)
    S_02 = S3(vis_02, time, freq, prms1)
    S_12 = S3(vis_12, time, freq, prms2)
    return S_01+S_02+S_12

def art_phase(phi0, r0, tau0, time, freq, noise, kdisp=0):
    fmin = np.min(freq)
    fmax = np.max(freq)
    phase = phi0 + 2*np.pi*(r0*time[:, np.newaxis] + tau0*(freq-freq[0])
                            + kdisp*(1/freq + (freq - fmin - fmax)/(fmin*fmax)))
    return phase + noise*np.random.normal(0,1,(len(time), len(freq)))

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
    fmin = np.min(freq)
    fmax = np.max(freq)
    vis_model = np.zeros_like(vis_global)
    for tint in range(intvals_t):
        tt1 = tint*ttt
        tt2 = tt1+ttt
        if (tint==intvals_t-1):
            tt2 = nt-1
        G_ff_inv = np.exp(-1j*(prms[tint, 0]
                               +2*np.pi*(prms[tint, 1]/tunit*(time[tt1:tt2+1, np.newaxis]-time[tt1])
                                         + prms[tint, 2]/funit*(freq-freq[0])
                                         + prms[tint, 3]*kunit*(1/freq + (freq - fmin - fmax)/(fmin*fmax)))
                              ))
        vis_cal[tt1:tt2+1] = vis_global[tt1:tt2+1]*G_ff_inv 
        vis_model[tt1:tt2+1] = G_ff_inv
    if model:
        return vis_model

########## LOAD DATA ##########
data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data    = table(data_MS)
time    = np.unique(data.getcol("TIME"))
t0      = time[0]
time   -= t0
nt      = len(time)
data.close()

spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
freq    = spectral_window[0]['CHAN_FREQ']
Dfrq    = freq - freq[0]
nf      = len(freq)
fmin    = np.min(freq)
fmax    = np.max(freq)
spectral_window.close()

########## PARAMETERS ##########
nscale  = 16
nant    = 3

#3 Baselines: 0 - 1 - 2
r       = np.array([0.0, 0.5e-3, 1e-3])  # 1e-3 = Hz 1 ms
tau     = np.array([0.0, 0.5e-7, 1e-7])    # 1e-9 = 1 ns
phi0    = np.array([0.0, -np.pi/3, 0.3])
kdisp   = np.array([0.0, -375, -125])*1e6

noise   = 0
lsm     = False
tunit   = 1e3
funit   = 1e9
kunit   = 1e6
intvals_t = 1
params  = np.zeros([nant, intvals_t, 4])
snr_bef = np.zeros((intvals_t, nant-1))
ttt     = nt//intvals_t - 1

dt = time[1] - time[0]           # time resolution [s]
df = freq[1] - freq[0]           # freq resolution [Hz]

delay   = fftshift(fftfreq(nscale*nf, df))  # in [mu s]

########## VISIBILITIES ##########
vis_global_01 = np.exp(1j*art_phase(np.angle(np.exp(1j*(phi0[1]-phi0[0]))), r[1] - r[0], tau[1]- tau[0], time, freq, noise, kdisp[1]-kdisp[0]))
vis_global_02 = np.exp(1j*art_phase(np.angle(np.exp(1j*(phi0[2]-phi0[0]))), r[2] - r[0], tau[2]- tau[0], time, freq, noise, kdisp[2]-kdisp[0]))
vis_global_12 = np.exp(1j*art_phase(np.angle(np.exp(1j*(phi0[2]-phi0[1]))), r[2] - r[1], tau[2]- tau[1], time, freq, noise, kdisp[2]-kdisp[1]))
vis_cal_01 = np.zeros_like(vis_global_01)
vis_cal_02 = np.zeros_like(vis_global_02)
vis_cal_12 = np.zeros_like(vis_global_12)

##########  Initial Guess (FFT) ##########
ddf = 10
for tint in range(intvals_t):
    tt1 = tint*ttt
    tt2 = tt1+ttt
    if (tint==intvals_t-1):
        tt2 = nt-1
    nt_sub = tt2+1-tt1

    time_new    = np.linspace(time[tt1], time[tt2], nt_sub)
    fringe_rate = fftshift(fftfreq(nscale*nt_sub, dt))  # in Hz

    vis    = vis_global_01[tt1:tt2+1]
    phi0_, r_max_idx, tau_max_idx, snr_, sgm_1 = FFT_guess(vis, nscale, nt_sub, nf)
    params[1, tint, 0] = phi0_
    params[1, tint, 1] = fringe_rate[r_max_idx]*tunit
    params[1, tint, 2] = delay[tau_max_idx]*funit
    snr_bef[tint, 0]   = snr_

    phaMOD = phi0_+M_PI2*np.add.outer(fringe_rate[r_max_idx]/tunit*time_new, delay[tau_max_idx]/funit*Dfrq [nf//2 - ddf:nf//2 + ddf])
    phaVIS = np.angle(vis[:, nf//2 - ddf:nf//2 + ddf])
    Sumdis = (phaVIS - phaMOD)/(M_PI2*(1/freq[nf//2 - ddf:nf//2 + ddf] + (freq[nf//2 - ddf:nf//2 + ddf] - fmin - fmax)/(fmin*fmax)) + FLT_EPSILON)
    params[1, tint, 3] = np.sum(Sumdis)/(2*ddf*nt_sub*kunit)

    vis    = vis_global_02[tt1:tt2+1]
    phi0_, r_max_idx, tau_max_idx, snr_, sgm_2 = FFT_guess(vis, nscale, nt_sub, nf)
    params[2, tint, 0] = phi0_
    params[2, tint, 1] = fringe_rate[r_max_idx]*tunit
    params[2, tint, 2] = delay[tau_max_idx]*funit
    snr_bef[tint, 1]   = snr_

    phaMOD = phi0_+M_PI2*np.add.outer(fringe_rate[r_max_idx]/tunit*time_new, delay[tau_max_idx]/funit*Dfrq [nf//2 - ddf:nf//2 + ddf])
    phaVIS = np.angle(vis[:, nf//2 - ddf:nf//2 + ddf])
    Sumdis = (phaVIS - phaMOD)/(M_PI2*(1/freq[nf//2 - ddf:nf//2 + ddf] + (freq[nf//2 - ddf:nf//2 + ddf] - fmin - fmax)/(fmin*fmax)) + FLT_EPSILON)
    params[2, tint, 3] = np.sum(Sumdis)/(2*ddf*nt_sub*kunit)

########## Least-Squares Minimization ##########
INI_TIME = tm.time()
# print(f"T: 01: {phi0[0]:.2e}\t{r[0]*tunit:.2e}\t{tau[0]*funit:.2e}\t{kdisp[0]/kunit:.2e} \t12: {phi0[1]:.2e}\t{r[1]*tunit:.2e}\t{tau[1]*funit:.2e}\t{kdisp[1]/kunit:.2e} \t02: {phi0[2]:.2e}\t{r[2]*tunit:.2e}\t{tau[2]*funit:.2e}\t{kdisp[2]/kunit:.2e}")
# print(f"C: 01: {params[0, 0, 0]:.2e}\t{params[0, 0, 1]:.2e}\t{params[0, 0, 2]:.2e}\t{params[0, 0, 3]:.2e} \t12: {params[1, 0, 0]:.2e}\t{params[1, 0, 1]:.2e}\t{params[1, 0, 2]:.2e}\t{params[1, 0, 3]:.2e} \t02: {params[2, 0, 0]:.2e}\t{params[2, 0, 1]:.2e}\t{params[2, 0, 2]:.2e}\t{params[2, 0, 3]:.2e}")
if lsm:
    lsm_imp = np.zeros(intvals_t)
    lsm_err = np.zeros(intvals_t)
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
        # low_bounds = []
        # up_bounds = []
        for i in range(nant):
            bounds.extend([(-np.pi, np.pi), (-1/dt*tunit, 1/dt*tunit), (-1/df*funit, 1/df*funit), (-np.inf, np.inf)])  # phi0, r, tau
            # low_bounds.extend([-np.pi,-1/dt*tunit,-1/df*funit])  # phi0, r, tau
            # up_bounds.extend([np.pi, 1/dt*tunit, 1/df*funit])  # phi0, r, tau
            
        result  = minimize(objective, x0, args=(vis_global_01[tt1:tt2+1], vis_global_02[tt1:tt2+1], vis_global_12[tt1:tt2+1], time_new, freq), method='L-BFGS-B', bounds=bounds, options={'maxiter':100})
        # result  = least_squares(objective, x0, args=(vis_global_01[tt1:tt2+1], vis_global_02[tt1:tt2+1], vis_global_12[tt1:tt2+1], time_new, freq), bounds=(low_bounds, up_bounds), max_nfev=100)
        # result  = differential_evolution(objective, x0=x0, bounds=bounds, args=(vis_global_01[tt1:tt2+1], vis_global_02[tt1:tt2+1], vis_global_12[tt1:tt2+1], time_new, freq), maxiter=10)

        params[:, tint] = result.x.reshape(3, 4)
        lsm_err[tint] = result.fun
        lsm_imp[tint] = 100*(1.-lsm_err[tint]/S0)


######### Calibrate phase #########
model_01 = calibrate(vis_global_01, vis_cal_01, params[0], params[1], time, freq, intvals_t, ttt, model=True)
model_02 = calibrate(vis_global_02, vis_cal_02, params[0], params[2], time, freq, intvals_t, ttt, model=True)
model_12 = calibrate(vis_global_12, vis_cal_12, params[1], params[2], time, freq, intvals_t, ttt, model=True)

#### Plot Results
print(f"L: 01: {params[0, 0, 0]:.2e}\t{params[0, 0, 1]:.2e}\t{params[0, 0, 2]:.2e}\t{params[0, 0, 3]:.2e} \t12: {params[1, 0, 0]:.2e}\t{params[1, 0, 1]:.2e}\t{params[1, 0, 2]:.2e}\t{params[1, 0, 3]:.2e} \t02: {params[2, 0, 0]:.2e}\t{params[2, 0, 1]:.2e}\t{params[2, 0, 2]:.2e}\t{params[2, 0, 3]:.2e}")
print("TIME: ", tm.time()-INI_TIME)

if lsm:
    for tint in range(intvals_t):
        print(f"Interval: {tint}\t Imp. LSM: {lsm_imp[tint]:.5f} % \t SNR: {np.mean(snr_bef[tint]):.3f}")

plot_phase(np.angle(vis_global_01), None, time, freq, len(time), len(freq), Baselinetitle="Original 01", showw=False)
plot_phase(np.angle(vis_global_02), None, time, freq, len(time), len(freq), Baselinetitle="Original 02", showw=False)
plot_phase(np.angle(vis_global_12), None, time, freq, len(time), len(freq), Baselinetitle="Original 12", showw=False)

plot_phase(np.angle(model_01), None, time, freq, len(time), len(freq), Baselinetitle="Model 01", showw=False)
plot_phase(np.angle(model_02), None, time, freq, len(time), len(freq), Baselinetitle="Model 02", showw=False)
plot_phase(np.angle(model_12), None, time, freq, len(time), len(freq), Baselinetitle="Model 12", showw=False)

plot_phase(np.angle(vis_cal_01), None, time, freq, len(time), len(freq), Baselinetitle="Calibrated 01", showw=False)
plot_phase(np.angle(vis_cal_02), None, time, freq, len(time), len(freq), Baselinetitle="Calibrated 02", showw=False)
plot_phase(np.angle(vis_cal_12), None, time, freq, len(time), len(freq), Baselinetitle="Calibrated 12", showw=True)