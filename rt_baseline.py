import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq

# Path to the MeasurementSet (MS)
data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

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

ant1 = 30
dant = 3
ntot = int((nant-ant1)/dant)
ff1  = 0
ff2  = int(len(freq)/1)-1
delay_max = np.zeros([ntot, 5, 16])
rate_max = np.zeros([ntot, 5, 16])
print(f"Ntot = {ntot}")
iant = 0
for ant2 in range(ant1+dant, nant, dant):
    iant += 1
    print(iant)
    t1  = data.query('ANTENNA1 == '+str(ant1) +' AND ANTENNA2 == '+str(ant2))   # do row selection
    for cc in range(5):
        pp = 4-cc
        intervals = 2**pp
        ttt = int(len(time)/intervals)-1
        for nnn in range(intervals):
            tt1 = nnn*ttt
            tt2 = tt1+ttt

            vis = t1.getcol('CALIBRATED_DATA')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])
            flg = t1.getcol('FLAG')[tt1:tt2+1, ff1:ff2+1, n_pol]   # (channel, pol [RR, RL, LR, LL])

            ####### FFT ######
            vis[flg] = 0.0

            nt_sub, nf_sub = vis.shape

            # 2D FFT
            F_sub = fftshift(fft2(vis, norm='ortho'))
            power_sub = np.abs(F_sub)

            # ---------------------
            # FFT axes
            # ---------------------
            time_new = np.linspace(time[tt1], time[tt2], nt_sub)
            dt = (time_new[1] - time_new[0])             # time resolution [s]

            freq_new = np.linspace(freq[ff1], freq[ff2], nf_sub)
            df = (freq_new[1] - freq_new[0])           # freq resolution [Hz]

            fringe_rate = fftshift(fftfreq(nt_sub, dt))  # in Hz
            delay       = fftshift(fftfreq(nf_sub, df))*1e6  # in [mu s]

            # ---------------------
            # Find argmax
            # ---------------------
            peak_idx = np.unravel_index(np.argmax(power_sub), power_sub.shape)
            r_max_idx, tau_max_idx = peak_idx

            r_max = fringe_rate[r_max_idx]   # Hz
            tau_max = delay[tau_max_idx]     # seconds

            delay_max[iant-1, pp, nnn] = tau_max*1e3
            rate_max[iant-1, pp, nnn]  = r_max*1e3
data.close()
np.save("delay_baseline_"+str(ant1)+"_"+str(dant)+".npy", delay_max)
np.save("rate_baseline_" +str(ant1)+"_"+str(dant)+".npy", rate_max)