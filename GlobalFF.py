import numpy as np
import gc
import datetime
import matplotlib.pyplot as plt
from casacore.tables     import table
from matplotlib.gridspec import GridSpec
from numpy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import minimize

########## 0 READING DATA ##########
class LOAD_MS:
    def __init__(self, ms_path, npol=0):
        self.ms_path = ms_path
        self.n_pol   = npol

        # Load and process time
        self.data  = table(self.ms_path, readonly=True)
        self.time  = np.unique(self.data.getcol("TIME"))
        self.t0    = self.time[0]
        self.time -= self.t0
        self.nt    = len(self.time)
        self.dt    = self.time[1]-self.time[0]

        # Load frequency info
        spectral_window = table(f"{ms_path}/SPECTRAL_WINDOW")
        self.freq = spectral_window[0]['CHAN_FREQ']
        self.nf   = len(self.freq)
        self.df   = self.freq[1] - self.freq[0]
        self.Dfrq = self.freq - self.freq[0]
        spectral_window.close()

        # Load antenna info
        antennas     = table(f"{ms_path}/ANTENNA")
        self.nameant = antennas.getcol("NAME")
        self.nant    = len(self.nameant)
        antennas.close()

        self.baseline_ok = np.ones([self.nant, self.nant], dtype=bool)

    def close(self):
        self.data.close()
    
    def load_baseline(self, ant1, ant2):
        if (ant1 >= ant2):
            print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
            exit()
        t1   = self.data.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')
        vis  = t1.getcol('DATA')[:, :, self.n_pol]
        flg  = t1.getcol('FLAG')[:, :, self.n_pol]
        t1.close()
        vis[flg] = 0.0
        return vis

    def flagged_baselines(self, flagged_threshold = 0.3):
        print("Filtering flagged baselines")
        self.flg_th = flagged_threshold
        ntot = self.nt*self.nf
        for ant1 in range(self.nant):
            for ant2 in range(ant1+1, self.nant):
                t1 = self.data.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}', columns="FLAG")
                t2 = self.data.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}', columns="UVW", limit=1)
                flg = t1.getcol('FLAG')[:, :, self.n_pol]
                uvw = t2.getcol('UVW')[0]
                t1.close()
                ok = ((np.sum(flg)/ntot) < self.flg_th) and (np.sqrt(uvw[0]**2 + uvw[1]**2 + uvw[2]**2)>1e5)
                self.baseline_ok[ant1, ant2] = ok
                self.baseline_ok[ant2, ant1] = ok
        print("Flagged data ratio successfully calculated!")

class FringeFitting:
    def __init__(self, ms, tints=8, nscale=8, tunit=1e3, funit=1e9, refant=0):
        """
        ms: an instance of a class (like LOAD_MS)
        nscale: padding factor for FFT
        tunit: time unit conversion (e.g., to ms)
        funit: freq unit conversion (e.g., to MHz or GHz)
        """
        self.nscale = nscale
        self.tunit  = tunit
        self.funit  = funit
        self.tints  = tints
        self.refant = refant
        self.time_edgs = np.arange(0, ms.nt, ms.nt//self.tints)
        self.time_edgs[-1] = ms.nt-1
        self.params = np.zeros((ms.nant, self.tints, 3))

    def SNR(self, ampl_FFT):
        """Estimate signal-to-noise ratio (basic max/mean)."""
        return np.max(ampl_FFT) / (np.mean(ampl_FFT) + 1e-10)

    def FFT_init_ij(self, ms, ant):
        """Compute fringe-rate, delay, and phase estimates over time-frequency blocks."""
        
        if (self.refant >= ant):
            print(f"Error: refant: {self.refant} must be smaller than Ant: {ant}")
            exit()

        print(f"Reading Baseline {self.refant}-{ant}")
        t1   = ms.data.query('ANTENNA1 == '+str(self.refant) +' AND ANTENNA2 == '+str(ant))   # do row selection
        vis_global = t1.getcol('DATA')[:, :, ms.n_pol]
        flg_global = t1.getcol('FLAG')[:, :, ms.n_pol]
        t1.close()
        vis_global[flg_global] = 0.0
        delay = fftshift(fftfreq(self.nscale * ms.nf, ms.df))

        for tint in range(self.tints):
            tt1 = self.time_edgs[tint]
            tt2 = self.time_edgs[tint+1]
            nt_loc  =  tt2 - tt1 + 1

            # 2D FFT
            F_sub = fftshift(fft2(vis_global[tt1:tt2+1], norm='ortho', s=[self.nscale*nt_loc, self.nscale*ms.nf]))
            ampli_loc = np.abs(F_sub)
            fringe_rate = fftshift(fftfreq(self.nscale * nt_loc, ms.dt))  # Hz

            # Find peak
            r_idx, tau_idx = np.unravel_index(np.argmax(ampli_loc), ampli_loc.shape)

            self.params[ant, tint, 0] = np.angle(F_sub[r_idx, tau_idx])
            self.params[ant, tint, 1] = fringe_rate[r_idx] * self.tunit
            self.params[ant, tint, 2] = delay[tau_idx] * self.funit

    def FFT_init_global(self, ms):
        print("Calculating Initial Guess")
        for iant in range(1, ms.nant):
            if ms.baseline_ok[self.refant, iant]:
                self.FFT_init_ij(ms, iant)
        print("Initial Guess set successfully")

    def S3(self, vis_loc, wgt_loc, time, Df, prms0, prms1):
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

        prms0[0]  = (prms0[0] +np.pi)%(2*np.pi) - np.pi
        prms1[0]  = (prms1[0] +np.pi)%(2*np.pi) - np.pi

        phi0_0= prms0[0]
        r_0   = prms0[1]
        tau_0 = prms0[2]

        phi0_1= prms1[0]
        r_1   = prms1[1]
        tau_1 = prms1[2]

        phi0  = (phi0_1-phi0_0+np.pi)%(2*np.pi) - np.pi
        r     = (r_1-r_0)/self.tunit
        tau   = (tau_1-tau_0)/self.funit

        Dt    = time[:, np.newaxis] - time[0]
        Eijk  = np.exp(1j*(phi0 + 2*np.pi*(r*Dt + tau*Df)))
        S2_t  = wgt_loc*np.abs(vis_loc-Eijk)**2
        S     = np.sum(S2_t)
        return S
    
    def GlobalCost(self, params, ms, vis, wgt, time_n, tint):
        """
        Cost function for least-squares optimization of fringe parameters across all baselines.

        Parameters:
            prms (ndarray): Flattened array of parameters [phi0, r, tau] for all antennas.
            vis_01, vis_02, vis_12 (ndarray): Visibilities for the three baselines.
            time, freq (ndarray): Time and frequency axes.

        Returns:
            float: Sum of squared residuals across all three baselines.
        """
        S = 0.0
        tt1 = self.time_edgs[tint]
        tt2 = self.time_edgs[tint+1]
        for iant in range(ms.nant):
            for jant in range(iant+1, ms.nant):
                if ms.baseline_ok[iant, jant]:
                    vis_loc = vis[tt1:tt2+1]
                    wgt_loc = wgt[tt1:tt2+1]
                    S += self.S3(vis_loc, wgt_loc, time_n, ms.Dfrq, params[3*iant:3*(iant+1)], params[3*jant:3*(jant+1)])
        return S
    
    def LSM_FF(self, ms, maxiter=50):
        print("LS Minimizations is starting...")
        for tint in range(self.tints):
            x0 = self.params[:, tint].flatten()
            
            tt1 = self.time_edgs[tint]
            tt2 = self.time_edgs[tint+1]
            time_ini = ms.t0+ms.time[tt1]
            time_end = ms.t0+ms.time[tt2]
            time_new = ms.time[tt1:tt2+1]-ms.time[tt1]
            t1   = ms.data.query(f'TIME >= {time_ini} && TIME < {time_end}')
            vis  = t1.getcol('DATA')[:, :, ms.n_pol]
            flg  = t1.getcol('FLAG')[:, :, ms.n_pol]
            wgt  = t1.getcol('WEIGHT_SPECTRUM')[:, :, ms.n_pol]
            t1.close()
            vis[flg] = 0.0
            print(f"LSM tint: {tint}")
            result  = minimize(self.GlobalCost, x0, args=(ms, vis, wgt, time_new, tint), method='L-BFGS-B', options={'maxiter':maxiter})
            self.params[:, tint] = result.x.reshape(ms.nant, 3)

            del vis, flg, wgt
            gc.collect()

        print(f"LSM is finished. Max. iteration is {maxiter}")

    def calibrate(self, ant1, ant2, ms):
        prms = self.params[ant2] - self.params[ant1]
        t1   = ms.data.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')
        vis_glob = t1.getcol('DATA')[:, :, ms.n_pol]
        flg_glob = t1.getcol('FLAG')[:, :, ms.n_pol]
        t1.close()
        vis_glob[flg_glob] = 0.0
        vis_cal = np.zeros_like(vis_glob)
        for tint in range(self.tints):
            tt1 = self.time_edgs[tint]
            tt2 = self.time_edgs[tint+1]
            G_ff_inv = np.exp(-1j*(prms[tint, 0]+2*np.pi*(prms[tint, 1]/self.tunit*(ms.time[tt1:tt2+1, np.newaxis]-ms.time[tt1])
                                                          + prms[tint, 2]/self.funit*(ms.freq-ms.freq[0]))))
            vis_cal[tt1:tt2+1] = vis_glob[tt1:tt2+1]*G_ff_inv
        return vis_cal
    
    def save_prmts(self, name='params'):
        np.save(f"{name}.npy", self.params)
    

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
    abs_glb_ph = np.abs(phase_global)

    # Create figure and grid layout
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Baseline: {BL}")
    gs  = GridSpec(3, 3, width_ratios=[8, 1, 1], height_ratios=[1, 1, 4],
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

    ax_main.imshow(phase_global.T, aspect='auto', origin='lower', vmax=np.pi, vmin=-np.pi,
                    extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
                    cmap='twilight')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    if showw:
        plt.show()
    
# Path to the MeasurementSet (MS)
data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

INI_TIME = datetime.datetime.now()
ms_data = LOAD_MS(data_MS)
ms_data.flagged_baselines()

print("TIME: ", (datetime.datetime.now()-INI_TIME).seconds)
plt.figure(figsize=(8, 6))
plt.imshow(ms_data.baseline_ok, origin='lower')
plt.xlabel("Antenna 2")
plt.ylabel("Antenna 1")
plt.colorbar(label='% Flagged Data')
plt.grid(False)
plt.show()

Global_FF = FringeFitting(ms_data, tints=8)
Global_FF.FFT_init_global(ms_data)
print("TIME: ", (datetime.datetime.now()-INI_TIME).seconds)

Global_FF.LSM_FF(ms_data, maxiter=10)
print("TIME: ", (datetime.datetime.now()-INI_TIME).seconds)

Global_FF.save_prmts()

ant1 = 70
ant2 = 72

vis_12  = ms_data.load_baseline(ant1, ant2)
vis_cal = Global_FF.calibrate(ant1, ant2, ms_data)

ms_data.close()

plot_phase(vis_12, ms_data.time, ms_data.freq, BL=f"{ms_data.nameant[ant1]}-{ms_data.nameant[ant2]}", showw=False)
plot_phase(vis_cal, ms_data.time, ms_data.freq, BL=f"{ms_data.nameant[ant1]}-{ms_data.nameant[ant2]}", showw=True)