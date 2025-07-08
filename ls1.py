import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from numpy.fft           import fftshift, fft2, fftfreq
from scipy.optimize import minimize

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

nscale   = 8
refant   = 0
reffreq0 = freq[refant]
refTime  = t0

data_loc = data.query(f"TIME >= {start_time} && TIME <= {end_time}")
data.close()
params   = np.zeros(3*nant)
nt_loc   = len(np.unique(data_loc.getcol("TIME")))

# Set initial guess
method = 2
ant1 = refant
for ant2 in range(ant1+1, nant):
    t1  = data_loc.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')   # do row selection
    vis_dat = t1.getcol('DATA')[:, n_pol]   # (channel, pol [RR, RL, LR, LL])
    if method==1:
        vis = vis_dat
    elif method==2:
        vis  = np.exp(1j*np.angle(vis_dat))
    else:
        print("Error: Method not defined")
        exit()

    flg = t1.getcol('FLAG')[:, n_pol]   # (channel, pol [RR, RL, LR, LL])
    vis[flg] = 0.0

    ####### FFT ######
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

    params[3*ant2+0] = phase_FT[r_max_idx, tau_max_idx]
    params[3*ant2+1] = fringe_rate[r_max_idx]  # Hz
    params[3*ant2+2] = delay[tau_max_idx]      # seconds

data.close()
annt = np.arange(1, nant)
plt.plot(annt, params[::3], ".", ms=5, c="k")
plt.xlim(annt[0], annt[-1])
plt.ylim(-np.pi, np.pi)
plt.ylabel("Phase")
plt.xlabel("Ant2")
plt.show()

plt.plot(annt, params[1::3]*1e3, ".", ms=5, c="k")
plt.xlim(annt[0], annt[-1])
plt.ylabel("Fringe rate [mHz]")
plt.xlabel("Ant2")
plt.show()

plt.plot(annt, params[2::3]*1e6, ".", ms=5, c="k")
plt.xlim(annt[0], annt[-1])
plt.ylabel(r"Delay $[\mu s]$")
plt.xlabel("Ant2")
plt.show()

def S3(data_loc, params):
    S = 0
    for ii in range(ncorr*nt_loc): # baselines and time (i,j,l)
        dataii = data_loc[ii]
        vis    = dataii['DATA'][:, n_pol]
        flg    = dataii['FLAG'][:, n_pol]
        vis[flg] = 0.0
        ant1  = dataii['ANTENNA1']
        ant2  = dataii['ANTENNA2']
        t1    = dataii['TIME']
        phi01 = params[3*ant1+0]
        tau1  = params[3*ant1+1]
        r1    = params[3*ant1+2]
        phi02 = params[3*ant2+0]
        tau2  = params[3*ant2+1]
        r2    = params[3*ant2+2]
        phi0  = phi02-phi01
        tau   = tau2-tau1
        r     = r2-r1
        Dt    = t1 - refTime
        Df    = freq-reffreq0

        Eijk  = np.exp(1j*(phi0 + 2*np.pi*r*Dt + 2*np.pi*tau*Df))
        Theo  = np.exp(1j*np.angle(vis))

        S2_t  = np.abs(Theo-Eijk)**2
        S    += np.sum(S2_t)
    return S


S3_ini = S3(data_loc, params)

def objective(params, data_loc):
    return S3(data_loc, params)

# Run the optimization
result = minimize(objective, params, args=(data_loc,), method='SLSQP', options={'maxiter':10})

# Access optimized parameters
optimized_params = result.x
final_value = result.fun
print(optimized_params)
print(final_value)
if result.success:
    print("Optimization succeeded.")
else:
    print("Optimization failed:", result.message)