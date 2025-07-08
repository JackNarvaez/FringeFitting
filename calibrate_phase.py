import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import minimize

def SNR(dataFFT):
    return np.max(dataFFT)/np.mean(dataFFT)

def S3(vis_loc, wgt_loc, time, freq, params):
    params[0]  = (params[0] +np.pi)%(2*np.pi) - np.pi
    phi0  = params[0]
    r     = params[1]/tunit
    tau   = params[2]/funit
    Dt    = time[:, np.newaxis] - time[0]
    Df    = freq - freq[0]
    Eijk  = np.exp(1j*(phi0 + 2*np.pi*(r*Dt + tau*Df)))
    S2_t  = wgt_loc*np.abs(vis_loc-Eijk)**2
    S     = np.sum(S2_t)
    return S

def objective(params, vis_loc, wgt_loc, time, freq):
    return S3(vis_loc, wgt_loc, time, freq, params)

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
nameant  = antennas.getcol("NAME")
nant     = len(nameant)
antennas.close()

nscale   = 8
ant1     = 44
ant2     = 70
lsm      = True
tunit    = 1e3
funit    = 1e9

if (ant1 >= ant2):
    print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
    exit()

t1  = data.query('ANTENNA1 == '+str(ant1) +' AND ANTENNA2 == '+str(ant2))   # do row selection
data.close()

vis_global = t1.getcol('DATA')[:, :, n_pol]
flg_global = t1.getcol('FLAG')[:, :, n_pol]
if lsm:
    wgt_global = t1.getcol('WEIGHT_SPECTRUM')[:, :, n_pol]
    wgt_global[flg_global] = 0.0
vis_global[flg_global] = 0.0
phase_global = np.angle(vis_global)

ff1 = 0
ff2 = int(len(freq)/1)-1

intvals_t = 8
intvals_f = 1
params    = np.zeros([intvals_t, intvals_f, 3])
ttt = int(len(time)/intvals_t)-1
fff = int(len(freq)/intvals_f)-1
time_edgs = np.zeros(intvals_t)
freq_edgs = np.zeros(intvals_f)
vis_cal = np.zeros([nt, nf], dtype='c16')

for tint in range(intvals_t):
    tt1 = tint*ttt
    tt2 = tt1+ttt
    if (tint==intvals_t-1):
        tt2 = nt-1
    time_edgs[tint] = time[tt2]
    for fint in range(intvals_f):
        ff1 = fint*fff
        ff2 = ff1+fff

        if (fint==intvals_f-1):
            ff2 = nf-1
        if (tint == 0):
            freq_edgs[fint] = freq[ff2]
        vis = vis_global[tt1:tt2+1, ff1:ff2+1]   # (channel, pol [RR, RL, LR, LL])

        ####### FFT ######
        nt_sub, nf_sub = vis.shape

        # 2D FFT
        F_sub = fftshift(fft2(vis, norm='ortho', s=[nscale*nt_sub, nscale*nf_sub]))
        power_sub = np.abs(F_sub)
        phase_sub = np.angle(F_sub)

        print(SNR(power_sub))

        # ---------------------
        # FFT axes
        # ---------------------
        time_new = np.linspace(time[tt1], time[tt2], nt_sub)
        dt = (time_new[1] - time_new[0])             # time resolution [s]

        freq_new = np.linspace(freq[ff1], freq[ff2], nf_sub)
        df = (freq_new[1] - freq_new[0])           # freq resolution [Hz]

        fringe_rate = fftshift(fftfreq(nscale*nt_sub, dt))  # in Hz
        delay       = fftshift(fftfreq(nscale*nf_sub, df))  # in [mu s]

        # ---------------------
        # Find argmax
        # ---------------------
        r_max_idx, tau_max_idx = np.unravel_index(np.argmax(power_sub), power_sub.shape)

        params[tint, fint, 0] = phase_sub[r_max_idx, tau_max_idx]
        params[tint, fint, 1] = fringe_rate[r_max_idx]*tunit
        params[tint, fint, 2] = delay[tau_max_idx]*funit

        if lsm:
            wgt = wgt_global[tt1:tt2+1, ff1:ff2+1]   # (channel, pol [RR, RL, LR, LL])
            params0 = params[tint, fint].copy()
            S0      = S3(vis, wgt, time_new, freq_new,params0)
            result  = minimize(objective, params[tint, fint], args=(vis, wgt, time_new, freq_new), method='L-BFGS-B', options={'maxiter':100})
            params[tint, fint] = result.x
            S1 = result.fun
            print(f"{params0[0]:.5f}\t\t{params0[1]:.5f}\t\t{params0[2]:.5f}")
            print(f"{params[tint, fint, 0]:.5f}\t\t{params[tint, fint, 1]:.5f}\t\t{params[tint, fint, 2]:.5f}")
            print(f"res: {100*(1.-S1/S0):.5f} %")
        
        #### Calibrate phase
        G_ff_inv = np.exp(-1j*(params[tint, fint, 0]+2*np.pi*(params[tint, fint, 1]/tunit*(time[tt1:tt2+1, np.newaxis]-time[tt1]) + params[tint, fint, 2]/funit*(freq[ff1:ff2+1]-freq[ff1]))))
        vis_cal[tt1:tt2+1, ff1:ff2+1] = vis*G_ff_inv

vis_cal[flg_global] = 0.0
phase_cal  = np.angle(vis_cal)
abs_glb_ph = np.abs(phase_global)
abs_glb_cl = np.abs(phase_cal)

# Create figure and grid layout
fig = plt.figure(figsize=(10, 6))
fig.suptitle(nameant[ant1] +"-" +nameant[ant2])
gs = GridSpec(3, 3, width_ratios=[8, 1, 1], height_ratios=[1, 1, 4],
              wspace=0.05, hspace=0.05)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.plot(time, np.std(abs_glb_ph, axis=1), ".", color='purple')
ax_top.set_xlim(time[0], time[-1])
ax_top.set_ylabel("STD")
ax_top.set_xticks([])

ax_top2 = fig.add_subplot(gs[1, 0])
ax_top2.plot(time, np.mean(abs_glb_ph, axis=1), ".", color='purple')
ax_top2.set_ylabel("MEAN")
ax_top2.set_xlim(time[0], time[-1])
ax_top2.set_ylim(-np.pi, np.pi)
ax_top2.set_xticks([])

ax_right = fig.add_subplot(gs[2, 2])
ax_right.plot(np.std(abs_glb_ph,axis=0), freq, ".", color='green')
ax_right.set_ylim(freq[0], freq[-1])
ax_right.set_xlabel("STD")
ax_right.set_yticks([])

ax_right2 = fig.add_subplot(gs[2, 1])
ax_right2.plot(np.mean(abs_glb_ph,axis=0), freq, ".", color='green')
ax_right2.set_xlabel("MEAN")
ax_right2.set_ylim(freq[0], freq[-1])
ax_right2.set_xlim(-np.pi, np.pi)
ax_right2.set_yticks([])

ax_main = fig.add_subplot(gs[2, 0])
im = plt.imshow(phase_global.T, aspect='auto', origin='lower',
                    extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")


# Create figure and grid layout
fig = plt.figure(figsize=(10, 6))
fig.suptitle(nameant[ant1] +"-" +nameant[ant2])
gs = GridSpec(3, 3, width_ratios=[8, 1, 1], height_ratios=[1, 1, 4],
              wspace=0.05, hspace=0.05)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.plot(time, np.std(abs_glb_cl, axis=1), ".", color='purple')
ax_top.set_xlim(time[0], time[-1])
ax_top.set_ylabel("STD")
ax_top.set_xticks([])

ax_top2 = fig.add_subplot(gs[1, 0])
ax_top2.plot(time, np.mean(abs_glb_cl, axis=1), ".", color='purple')
for t in time_edgs: {ax_top2.axvline(t, ls="-", c="k", alpha=0.8)}
ax_top2.set_ylabel("MEAN")
ax_top2.set_xlim(time[0], time[-1])
ax_top2.set_ylim(-np.pi, np.pi)
ax_top2.set_xticks([])

ax_right = fig.add_subplot(gs[2, 2])
ax_right.plot(np.std(abs_glb_cl,axis=0), freq, ".", color='green')
ax_right.set_ylim(freq[0], freq[-1])
ax_right.set_xlabel("STD")
ax_right.set_yticks([])

ax_right2 = fig.add_subplot(gs[2, 1])
ax_right2.plot(np.mean(abs_glb_cl,axis=0), freq, ".", color='green')
for f in freq_edgs: {ax_right2.axhline(f, ls="-", c="k", alpha=0.8)}
ax_right2.set_xlabel("MEAN")
ax_right2.set_ylim(freq[0], freq[-1])
ax_right2.set_xlim(-np.pi, np.pi)
ax_right2.set_yticks([])

ax_main = fig.add_subplot(gs[2, 0])
im = plt.imshow(phase_cal.T, aspect='auto', origin='lower',
                    extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')
for t in time_edgs: {plt.axvline(t, ls="-", lw=1, c="w", alpha=0.5)}
for f in freq_edgs: {plt.axhline(f/1e6, ls="-", lw=1, c="w", alpha=0.5)}
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
plt.show()