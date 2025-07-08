import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from numpy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import least_squares

data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data = table(data_MS, ack=False, readonly=True)
time = data.getcol("TIME")

start_time = time.min()
end_time = start_time + 1800  # 1 hour = 3600 seconds

data    = table(data_MS)
n_rows  = int(len(data))
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
ncorr    = int(nant*(nant-1)/2)

refant   = 0
reffreq0 = freq[refant]
refTime  = t0

data_loc = data.query(f"TIME >= {start_time} && TIME <= {end_time}")
data.close()
params   = np.zeros(3*nant-3)
nt_loc   = len(np.unique(data_loc.getcol("TIME")))

# Set initial guess
ant1 = refant
for ant2 in range(ant1+1, nant):
    t1  = data_loc.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')   # do row selection
    vis_dat = t1.getcol('CALIBRATED_DATA')[:, n_pol]   # (channel, pol [RR, RL, LR, LL])
    vis_mod = t1.getcol('MODEL_DATA')[:, n_pol]   # (channel, pol [RR, RL, LR, LL])
    flg = t1.getcol('FLAG')[:, n_pol]   # (channel, pol [RR, RL, LR, LL])
    
    ####### FFT ######
    vis  = np.exp(1j*(np.angle(vis_dat)-np.angle(vis_mod)))
    vis[flg] = 0.0

    nnt, nnf = vis.shape
    dt = time[1] - time[0]       # time resolution [s]
    df = freq[1] - freq[0]       # freq resolution [Hz]
    F_s = fftshift(fft2(vis, norm="ortho"))
    fringe_rate = fftshift(fftfreq(nnt, dt))     # [Hz]
    delay       = fftshift(fftfreq(nnf, df))     # [mu s]

    power_FT   = np.abs(F_s)
    phase_FT   = np.angle(F_s)

    peak_idx = np.unravel_index(np.argmax(power_FT), power_FT.shape)
    r_max_idx, tau_max_idx = peak_idx

    params[3*(ant2-1)+0] = phase_FT[r_max_idx, tau_max_idx]
    params[3*(ant2-1)+1] = fringe_rate[r_max_idx]  # Hz
    params[3*(ant2-1)+2] = delay[tau_max_idx]      # seconds

data.close()

annt = np.arange(1, nant)
plt.plot(annt, params[::3], c="k")
plt.xlim(annt[0], annt[-1])
plt.ylim(-np.pi, np.pi)
plt.ylabel("Phase")
plt.show()

plt.plot(annt, params[1::3]*1e3, c="k")
plt.xlim(annt[0], annt[-1])
plt.ylabel("Fringe rate [mHz]")
plt.show()

plt.plot(annt, params[2::3]*1e6, c="k")
plt.xlim(annt[0], annt[-1])
plt.ylabel(r"Delay $[\mu s]$")
plt.show()

def S3(params, data_loc):
    print(params)
    S = np.zeros(2*ncorr)
    for ii in range(ncorr*nt_loc): # baselines and time (i,j,l)
        dataii = data_loc[ii]
        vis    = dataii['DATA'][:, n_pol]
        v_mdl  = dataii['MODEL_DATA'][:, n_pol]
        flg    = dataii['FLAG'][:, n_pol]
        vis[flg] = 0.0
        ant1  = dataii['ANTENNA1']
        ant2  = dataii['ANTENNA2']
        t1    = dataii['TIME']
        phi01 = params[3*(ant1-1)]
        tau1  = params[3*(ant1-1)+1]
        r1    = params[3*(ant1-1)+2]
        phi02 = params[3*(ant2-1)]
        tau2  = params[3*(ant2-1)+1]
        r2    = params[3*(ant2-1)+2]
        phi0  = phi02-phi01
        tau   = tau2-tau1
        r     = r2-r1

        Dt    = t1 - refTime

        Eijk  = np.exp(1j*(phi0 + 2*np.pi*r*Dt + 2*np.pi*tau*(freq-freq[0])))
        Theo  = np.exp(1j*(np.angle(vis)-np.angle(v_mdl)))

        S2_t  = (Theo-Eijk)**2
        S[ii%ncorr] += np.sum(S2_t.real)
        S[ii%ncorr + ncorr] += np.sum(S2_t.imag)
    return S

# Run the optimization
result = least_squares(S3, params, args=(data_loc,), method='trf', verbose=2)

# Get optimized parameters
optimized_params = result.x
final_residual_norm = result.cost  # Sum of squares