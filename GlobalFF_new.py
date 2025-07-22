import numpy as np
import gc
import time as tm
import matplotlib.pyplot as plt
from casacore.tables     import table, makearrcoldesc, maketabdesc
from matplotlib.gridspec import GridSpec
from scipy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import minimize, least_squares, differential_evolution

INI_TIME = tm.time()
FLT_EPSILON = np.finfo(float).eps
M_PI = np.pi
M_PI2 = 2*np.pi
M_PIP2 = np.pi/2

########## 0 READING DATA ##########
def wrap_phase(phase):
    """Fast phase wrapping"""
    return ((phase + M_PI) % (M_PI2)) - M_PI

def snr_aips(peak, sumw_, sumww_, xcount_):
    if (peak > 0.999*sumw_):
        peak = 0.999*sumw_
        print("peak > 0.999*sumw_")

    if (abs(sumw_)<FLT_EPSILON):
        cwt = 0
        print("WEIGHT SUM ~ 0")
    else:
        x = M_PIP2 *peak/sumw_
        # The magic numbers in the following formula are from AIPS FRING
        cwt = (np.tan(x)**1.163) * np.sqrt(sumw_/np.sqrt(sumww_/xcount_))
    return cwt

def SNR(ampl_FFT):
    """Estimate signal-to-noise ratio (basic max/mean)."""
    return np.max(ampl_FFT) / (np.mean(ampl_FFT) + 1e-10)

def baseline_ij(ant1, ant2):
    if (ant2 > ant1):
        return (ant2*(ant2-1))//2 + ant1
    elif (ant1 > ant2):
        return (ant1*(ant1-1))//2 + ant2
    else:
        print("ERROR: ANT1 == ANT2")
        return None

def plot_phase(phase, wgt_glb, time, freq, nbins_time, nbins_freq, nbins_phase = 50, Baselinetitle="", showw=False):
        # Create figure and grid layout
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f"Baseline: {Baselinetitle}")
        gs  = GridSpec(3, 3, width_ratios=[8, 1, 1], height_ratios=[1, 1, 4],
                    wspace=0.05, hspace=0.05)

        ax_top = fig.add_subplot(gs[0, 0])
        ax_top.plot(time, np.std(phase, axis=1), ".", color='purple')
        ax_top.set_xlim(time[0], time[-1])
        ax_top.set_ylabel("STD")
        ax_top.set_xticks([])

        ax_top2     = fig.add_subplot(gs[1, 0])
        time_rep    = np.repeat(time[:, np.newaxis], phase.shape[1], axis=1).flatten()

        if wgt_glb is not None:
            wgt = wgt_glb.flatten()
        else:
            wgt=None

        ax_top2.hist2d(
            time_rep, phase.flatten(),
            weights = wgt,
            bins=[nbins_time, nbins_phase],
            range=[[time[0], time[-1]], [-M_PI, M_PI]],
            cmap='viridis', cmin=1
        )

        ax_top2.set_ylabel("Phase")
        ax_top2.set_xlim(time[0], time[-1])
        ax_top2.set_ylim(-M_PI, M_PI)
        ax_top2.set_xticks([])

        ax_right = fig.add_subplot(gs[2, 2])
        ax_right.plot(np.std(phase,axis=0), freq, ".", color='green')
        ax_right.set_ylim(freq[0], freq[-1])
        ax_right.set_xlabel("STD")
        ax_right.set_yticks([])

        ax_right2 = fig.add_subplot(gs[2, 1])
        freq_rep    = np.repeat(freq[np.newaxis, :], phase.shape[0], axis=0).flatten()
        ax_right2.hist2d(
            phase.flatten(), freq_rep,
            weights = wgt,
            bins=[nbins_phase, nbins_freq],
            range=[[-M_PI, M_PI], [freq[0], freq[-1]]],
            cmap='viridis', cmin=1
        )

        ax_right2.set_xlabel("Phase")
        ax_right2.set_ylim(freq[0], freq[-1])
        ax_right2.set_xlim(-M_PI, M_PI)
        ax_right2.set_yticks([])

        ax_main = fig.add_subplot(gs[2, 0])

        ax_main.imshow(phase.T, aspect='auto', origin='lower', vmax=M_PI, vmin=-M_PI,
                        extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
                        cmap='twilight')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
        if showw:
            plt.show()

class LOAD_MS:
    def __init__(self, ms_path, npol=0, refant=0):
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
        self.f0   = np.mean(self.freq)
        self.df   = self.freq[1] - self.freq[0]
        self.Dfrq = self.freq - self.freq[0]
        spectral_window.close()

        # Load antenna info
        antennas     = table(f"{ms_path}/ANTENNA")
        self.nameant = antennas.getcol("NAME")
        self.nant    = len(self.nameant)
        self.n_bls   = self.nant*(self.nant-1)//2
        antennas.close()

        self.refant = refant
        self.baseline_ok = np.zeros(self.n_bls, dtype=bool)

    def close(self):
        self.data.close()
    
    def load_baseline(self, ant1, ant2, npol=0, Data=True, Wgt=True, Cal=False, Model=False):
        if (ant1 >= ant2):
            print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
            exit()
        t1   = self.data.query(f'ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')
        flg  = t1.getcol('FLAG')[:, :, npol]

        results = []
        if Data:
            vis  = t1.getcol('DATA')[:, :, npol]
            vis[flg] = 0.0
            results.append(vis)
        if Wgt:
            wgt = t1.getcol('WEIGHT_SPECTRUM')[:, :, npol]
            wgt[flg] = 0.0
            results.append(wgt)
        if Cal:
            cal = t1.getcol('CALIBRATED_DATA')[:, :, npol]
            cal[flg] = 0.0
            results.append(cal)
        if Model:
            mod = t1.getcol('CALIBRATED_MODEL')
            mod[flg] = 0.0
            results.append(mod)
        t1.close()

        return tuple(results)

    def flag_baselines(self, tini, tend, flagged_threshold = 0.3, uvrange = 7e4):
        print("\n ------- FLAGGING DATA ------- ")
        print(f"flagged_threshold = {flagged_threshold:.2f}")
        print(f"uvrange_threshold = {uvrange:.2f}")
        self.flg_th = flagged_threshold
        self.uvw_th = uvrange
        ntot = self.nf*(tend-tini)
        if not isinstance(self.n_pol, int):
            ntot *= len(self.n_pol)
        t_offset = tini*self.n_bls
        t_limit  = (tend-tini)*self.n_bls
        Tab = self.data.query(offset=t_offset, limit=t_limit, columns="FLAG,UVW")
        uvw = Tab.getcol('UVW')[:]
        flg = Tab.getcol('FLAG')[:, :, self.n_pol]
        Tab.close()
        temp_sum = np.zeros(self.n_bls)
        for irow in range(t_limit):
            bij = irow % self.n_bls;
            temp_sum[bij] += np.sum(flg[irow])

        for iant in range(self.nant):
            for jant in range(self.nant):
                if iant == jant:
                    continue
                bij   = baseline_ij(iant, jant)
                cond1 = np.sqrt(uvw[bij, 0]**2 + uvw[bij, 1]**2 + uvw[bij, 2]**2) > self.uvw_th
                cond2 = (iant == self.refant or jant==self.refant)
                if cond1 or cond2:
                    ok = temp_sum[bij]/ntot < self.flg_th
                    self.baseline_ok[bij] = ok

        print("------------------------------\n")

    def load_data(self, tini, tend, corrcomb=0):
        tloc = tend-tini
        ntot = self.nf*tloc
        t_offset = tini*self.n_bls
        t_limit  = tloc*self.n_bls

        print("\n ------- LOADING DATA ------- ")
        Tab = self.data.query(offset=t_offset, limit=t_limit, columns="DATA,FLAG,WEIGHT_SPECTRUM")
        temp_vis = Tab.getcol('DATA')[:, :, self.n_pol]
        temp_flg = Tab.getcol('FLAG')[:, :, self.n_pol]
        temp_wgt = Tab.getcol('WEIGHT_SPECTRUM')[:, :, self.n_pol]
        Tab.close()

        # Combine correlations
        if corrcomb==0:
            print("Polarization: ", self.n_pol)
        elif corrcomb==1:
            print("Combining correlations: weighted-average")
            temp_flg = np.all(temp_flg, axis=2)
            temp_vis = np.sum(temp_vis*temp_wgt, axis=2)/np.sum(temp_wgt, axis=2)
            temp_wgt = np.mean(temp_wgt, axis=2)
        elif corrcomb==2:
            print("Combining correlations: Stokes I")
            temp_flg = np.all(temp_flg, axis=2)
            temp_vis = 0.5*np.sum(temp_vis, axis=2)
            temp_wgt = np.sum(temp_wgt, axis=2)
        else:
            print(f"corrcomb={corrcomb} is not defined.")
            exit()

        self.vis = np.zeros_like(temp_vis)
        self.wgt = np.zeros_like(temp_wgt)
        flg      = np.zeros_like(temp_flg)

        print("Ordering data. ")

        for bl in range(self.n_bls):
            if self.baseline_ok[bl]:
                for tt in range(tloc):
                    orig_row = tt * self.n_bls + bl
                    new_row  = bl * tloc + tt
                    self.vis[new_row] = np.exp(1j*np.angle(temp_vis[orig_row]))
                    self.wgt[new_row] = temp_wgt[orig_row]
                    flg[new_row]      = temp_flg[orig_row]
        
        self.vis[flg] = 0.0
        self.wgt[flg] = 0.0
        
        print("------------------------------\n")

        del temp_vis, temp_flg, temp_wgt, flg
        gc.collect()

    def plot_phase_local(self, ant1, ant2, tini, tend, vis_cal=None, vis_wgt=None, showw=False):

        baseline = (ant2*(ant2-1))//2 + ant1
        tloc = tend-tini
        time_loc = self.time[tini:tend]
        if (vis_cal is None):
            phase_global = np.angle(self.vis[baseline*tloc:(baseline+1)*tloc])
            wgt_global = self.wgt[baseline*tloc:(baseline+1)*tloc]
            extra = "[Original]"
        else:
            phase_global = np.angle(vis_cal)
            if (vis_wgt is None):
                wgt_global = self.wgt[baseline*tloc:(baseline+1)*tloc]
            else:
                wgt_global   = vis_wgt
            extra = ""
        baselinetitle = f"{self.nameant[ant1]} - {self.nameant[ant2]} {extra}"
        plot_phase(phase_global, wgt_global, time_loc, self.freq,\
                   nbins_time=len(time_loc), nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle, showw=showw)

    def plot_phase_global(self, ant1, ant2, npol):

        vis_global, wgt_global, cal_global, mod_phase = self.load_baseline(ant1, ant2, npol=npol, Data=True, Wgt=True, Cal=True, Model=True)
        vis_phase = np.angle(vis_global)
        cal_phase = np.angle(cal_global)
        baselinetitle = f"{self.nameant[ant1]} - {self.nameant[ant2]}"
        plot_phase(vis_phase, wgt_global, self.time, self.freq,\
                   nbins_time=self.nt, nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle + " [Original]", showw=False)
        plot_phase(cal_phase, wgt_global, self.time, self.freq,\
                   nbins_time=self.nt, nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle + " [Calibrated]", showw=False)
        plot_phase(mod_phase, None, self.time, self.freq,\
                   nbins_time=self.nt, nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle + " [Model]", showw=True)


class FringeFitting:
    def __init__(self, ms, tints=8, nscale=8, tunit=1e3, funit=1e9):
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
        self.time_edgs = np.arange(0, ms.nt, ms.nt//self.tints)
        self.time_edgs[-1] = ms.nt
        self.params = np.zeros((ms.nant, 3))
        self.sumw_  = np.zeros((ms.nant))
        self.snr_   = np.zeros((ms.nant))
        self.snr1_  = np.zeros((ms.nant))

        self.refant = ms.refant

        # Constrains
        self.min_delay = self.funit * 0.5/(ms.freq[-1]-ms.freq[0])
        self.max_delay = self.funit * 0.5/ms.df
        self.min_rate  = self.tunit * 0.5/(ms.time[self.time_edgs[1]] - ms.time[self.time_edgs[0]])
        self.max_rate  = self.tunit * 0.5/ms.dt

        print(f"Reference Antenna: {ms.nameant[self.refant]}")

    def FFT_init_ij_old(self, ms, ant, nt_loc, fringe_rate, delay):
        """Compute fringe-rate, delay, and phase estimates over time-frequency blocks."""

        ant1, ant2 = (ant, self.refant) if self.refant > ant else (self.refant, ant)
        sgn = -1 if ant < self.refant else 1

        bl_ij = ant2*(ant2-1)//2 + ant1

        if ms.baseline_ok[bl_ij]:

            tt1 = bl_ij*nt_loc
            tt2 = tt1 + nt_loc

            # 2D FFT
            F_sub = fftshift(fft2(ms.vis[tt1:tt2], norm='ortho', s=[self.nscale*nt_loc, self.nscale*ms.nf]))
            ampli_loc = np.abs(F_sub)

            # Find peak
            amp_loc_sh = ampli_loc.shape
            r_idx, tau_idx = np.unravel_index(np.argmax(ampli_loc), amp_loc_sh)

            self.params[ant, 0] = sgn*np.angle(F_sub[r_idx, tau_idx])

            rate_ini  = sgn*fringe_rate[r_idx] * self.tunit
            delay_ini = sgn*delay[tau_idx] * self.funit

            sgn_r = -1 if rate_ini < 0 else 1
            sgn_d = -1 if delay_ini < 0 else 1

            if np.abs(rate_ini) < self.min_rate:
                rate_ini = sgn_r*self.min_rate
            elif np.abs(rate_ini) > self.max_rate:
                rate_ini = sgn_r*self.max_rate

            if np.abs(delay_ini) < self.min_delay:
                delay_ini = sgn_d*self.min_delay
            elif np.abs(delay_ini) > self.max_delay:
                delay_ini = sgn_d*self.max_delay
            self.sumw_[ant] = np.sum(ms.wgt[tt1:tt2])
            sumww_ = np.sum(ms.wgt[tt1:tt2]**2)
            xcount_ = amp_loc_sh[0]*amp_loc_sh[1]

            self.snr_[ant] = snr_aips(np.abs(F_sub[r_idx, tau_idx]), self.sumw_[ant], sumww_, xcount_)
            self.snr1_[ant] = SNR(ampli_loc)
        
        else:
            rate_ini  = sgn*self.min_rate
            delay_ini = sgn*self.min_delay

        self.params[ant, 1] = rate_ini   
        self.params[ant, 2] = delay_ini

        print(f"Baseline: {ant1}-{ant2} \t phase: {self.params[ant, 0]:.2f} rad \t rate: {self.params[ant, 1]:.2f} mHz \t delay: {self.params[ant, 2]:.2f} ns \t snr_mean {self.snr1_[ant]:.2f}\t snr_aips: {self.snr_[ant]:.2f}")

    def FFT_init_ij(self, ms, ant, nt_loc, fringe_rate, delay):
        ant1, ant2 = (ant, self.refant) if self.refant > ant else (self.refant, ant)
        sgn = -1 if ant < self.refant else 1

        bl_ij = ant2*(ant2-1)//2 + ant1

                    # FFT-derived limits
        df = ms.df
        nf_new = ms.nf * self.nscale
        max_delay = 0.5 / df
        min_delay = -0.5 / df
        bw = df * nf_new
        dt = ms.dt
        nt_new = nt_loc * self.nscale
        f0 = ms.f0
        max_rate = 0.5 / (dt * f0)
        min_rate = -0.5 / (dt * f0)
        width = dt * nt_new * f0

        if ms.baseline_ok[bl_ij]:
            tt1 = bl_ij * nt_loc
            tt2 = tt1 + nt_loc

            # FFT
            F_sub = fftshift(fft2(ms.vis[tt1:tt2], norm='ortho', s=[self.nscale * nt_loc, self.nscale * ms.nf]))
            ampli_loc = np.abs(F_sub)

            # Convert to bin indices (FFT grid)
            i0 = int(min_delay * bw)
            i1 = int(max_delay * bw)
            if i0 > i1:
                i0, i1 = i1, i0
            if i1 == i0:
                i1 += 1
            i0_shifted = (i0 + nf_new // 2) % nf_new
            i1_shifted = (i1 + nf_new // 2) % nf_new

            j0 = int(min_rate * width)
            j1 = int(max_rate * width)
            if j0 > j1:
                j0, j1 = j1, j0
            if j1 == j0:
                j1 += 1
            j0_shifted = (j0 + nt_new // 2) % nt_new
            j1_shifted = (j1 + nt_new // 2) % nt_new

            if i0_shifted < i1_shifted:
                delay_bins = slice(i0_shifted, i1_shifted)
            else:
                delay_bins = np.r_[np.arange(i0_shifted, nf_new), np.arange(0, i1_shifted)]

            if j0_shifted < j1_shifted:
                rate_bins = slice(j0_shifted, j1_shifted)
            else:
                rate_bins = np.r_[np.arange(j0_shifted, nt_new), np.arange(0, j1_shifted)]

            # Extract and search in region
            ampli_region = ampli_loc[np.ix_(rate_bins, delay_bins)]

            local_idx = np.argmax(ampli_region)
            local_r_idx, local_tau_idx = np.unravel_index(local_idx, ampli_region.shape)
            r_idx = (j0 + local_r_idx) % nt_new
            tau_idx = (i0 + local_tau_idx) % nf_new

            # Phase at peak
            self.params[ant, 0] = sgn * np.angle(F_sub[r_idx, tau_idx])

            rate_ini = sgn * fringe_rate[r_idx] * self.tunit
            delay_ini = sgn * delay[tau_idx] * self.funit

            self.sumw_[ant] = np.sum(ms.wgt[tt1:tt2])
            sumww_ = np.sum(ms.wgt[tt1:tt2] ** 2)
            xcount_ = ampli_region.size

            self.snr_[ant] = snr_aips(np.abs(F_sub[r_idx, tau_idx]), self.sumw_[ant], sumww_, xcount_)
            self.snr1_[ant] = SNR(ampli_region)
        else:
            rate_ini = sgn * min_rate
            delay_ini = sgn * min_delay

        self.params[ant, 1] = rate_ini
        self.params[ant, 2] = delay_ini

        print(f"Baseline: {ant1}-{ant2} \t phase: {self.params[ant, 0]:.2f} rad \t rate: {self.params[ant, 1]:.2f} Hz \t delay: {self.params[ant, 2]:.2f} s \t snr_mean {self.snr1_[ant]:.2f}\t snr_aips: {self.snr_[ant]:.2f}")

    def FFT_init_global(self, ms, tint):
        print("\n ------- INITIAL GUESS ------- ")
        print("Constrains")
        print(f"RATE: min: {self.min_rate:.2f} mHz \t max: {self.max_rate:.2f} mHz")
        print(f"DELAY: min: {self.min_delay:.2f} ns \t max: {self.max_delay:.2f} ns")
        print("------------------------------\n")
        nt_loc      =  self.time_edgs[tint+1]-self.time_edgs[tint]
        fringe_rate = fftshift(fftfreq(self.nscale * nt_loc, ms.dt))  # Hz
        delay       = fftshift(fftfreq(self.nscale * ms.nf, ms.df))
        for iant in range(ms.nant):
            if (self.refant == iant):
                continue
            self.FFT_init_ij(ms, iant, nt_loc, fringe_rate, delay)
        print("Initial Guess set successfully\n")

    def S3(self, vis_loc, wgt_loc, Dt, Df, prms):
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
        rDt   = (prms[1]/self.tunit)*Dt
        tauDf = (prms[2]/self.funit)*Df

        Eijk = np.exp(1j*(phi0 + M_PIP2*np.add.outer(rDt,tauDf)))
        S2_t = np.abs(vis_loc-Eijk)**2
        S    = np.sum(wgt_loc*S2_t)
        return S

    def GlobalCost(self, params, ms, nt_loc, time_n):
        """
        Cost function for least-squares optimization of fringe parameters across all baselines.

        Parameters:
            prms (ndarray): Flattened array of parameters [phi0, r, tau] for all antennas.
            vis_01, vis_02, vis_12 (ndarray): Visibilities for the three baselines.
            time, freq (ndarray): Time and frequency axes.

        Returns:
            float: Sum of squared residuals across all three baselines.
        """

        x0 = np.zeros(3*ms.n_bls)
        for iant in range(ms.nant):
            prmsi = params[3*iant:3*iant+3]
            for jant in range(iant+1, ms.nant):
                bl_ij = (jant*(jant-1))//2 + iant
                if not ms.baseline_ok[bl_ij]:
                    continue
                x0[3*bl_ij:3*bl_ij+3] = params[3*jant:3*jant+3] - prmsi

        S = 0.0
        for bl_ij in range(ms.n_bls):
            if not ms.baseline_ok[bl_ij]:
                continue
            tt1 = bl_ij*nt_loc
            tt2 = tt1 + nt_loc
            vis_loc = ms.vis[tt1:tt2]
            wgt_loc = ms.wgt[tt1:tt2]
            prm_loc = x0[3*bl_ij:3*bl_ij+3]
            S += self.S3(vis_loc, wgt_loc, time_n, ms.Dfrq, prm_loc)
            # S = max(S1, S)
        return S

    def LSM_FF(self, ms, tint, maxiter=50):
        print("\n ------- GLOBAL LSM ------- ")
        
        low_bounds = []
        upp_bounds = []
        for _ in range(ms.nant):
            low_bounds.extend([-M_PI,-self.max_rate, -self.max_delay])  # phi0, r, tau
            upp_bounds.extend([ M_PI, self.max_rate,  self.max_delay])  # phi0, r, tau

        x0   = self.params.flatten()
        tini = self.time_edgs[tint]
        tend = self.time_edgs[tint+1]
        nt_loc   =  tend-tini
        time_new = ms.time[tini:tend]-ms.time[tini]
        result  = least_squares(self.GlobalCost, x0, bounds=(low_bounds, upp_bounds), args=(ms, nt_loc, time_new), max_nfev=maxiter, verbose=2)
        print(f"residual: {result.fun[0]:.2f}")
        self.params = result.x.reshape(ms.nant, 3)

        print(f"LSM is finished. Max. iteration is {maxiter}")
        print("------------------------------\n")


    def calibrate(self, tini, tend, ms, tint, model=False):
        tloc = tend-tini
        t_offset = tini*ms.n_bls
        t_limit  = tloc*ms.n_bls
        
        print("\n ------- DATA CALIBRATION ------- ")
        Tab = ms.data.query(offset=t_offset, limit=t_limit, columns="DATA,FLAG")
        temp_vis = Tab.getcol('DATA')
        temp_flg = Tab.getcol('FLAG')
        n_rows = temp_vis.shape[0]
        Tab.close()

        # Ensure 'CALIBRATED_DATA' column exists
        Tab_cal  = table(ms.ms_path, readonly=False)
        if False:#(tint == 0):
            if 'CALIBRATED_DATA' in Tab_cal.colnames():
                Tab_cal.removecols('CALIBRATED_DATA')
            if 'CALIBRATED_MODEL' in Tab_cal.colnames():
                Tab_cal.removecols('CALIBRATED_MODEL')

        if 'CALIBRATED_DATA' not in Tab_cal.colnames():
            data_desc = Tab_cal.getcoldesc('DATA')
            data_desc['columnname'] = 'CALIBRATED_DATA'
            Tab_cal.addcols({'CALIBRATED_DATA': data_desc})
        if 'CALIBRATED_MODEL' not in Tab_cal.colnames():
            desc = makearrcoldesc(columnname='CALIBRATED_MODEL',
                          value=0.,
                          valuetype='float32',
                          ndim=1,
                          shape=[ms.nf])  # one phase per channel
            tabdesc = maketabdesc(desc)
            Tab_cal.addcols(tabdesc)

        calibrated_vis = temp_vis.copy()
        calibrated_mod = np.zeros((n_rows, ms.nf), dtype=np.float32)
        print("Start calibration...")
        for tt in range(tloc):
            for iant in range(ms.nant):
                prmsi = self.params[iant]
                for jant in range(iant+1, ms.nant):
                    bl_ij = baseline_ij(iant, jant)
                    if ~ms.baseline_ok[bl_ij]:
                        continue
                    row_idx = bl_ij + tt*ms.n_bls
                    prms = self.params[jant] - prmsi
                    phase_mod = prms[0]+M_PI2*(prms[1]/self.tunit*(ms.time[tt]-ms.time[tini])
                                             + prms[2]/self.funit*(ms.freq-ms.freq[0]))
                    G_ff_inv = np.exp(-1j*phase_mod)
                    for ii in range(temp_vis.shape[-1]):
                        calibrated_vis[row_idx, :, ii] = temp_vis[row_idx, :, ii]*G_ff_inv
                    calibrated_mod[row_idx] = np.angle(G_ff_inv)

        Tab_cal.putcol('CALIBRATED_DATA', calibrated_vis, startrow=t_offset, nrow=t_limit)
        Tab_cal.putcol('CALIBRATED_MODEL', calibrated_mod, startrow=t_offset, nrow=t_limit)
 
        Tab_cal.close()

        del temp_vis, temp_flg, calibrated_vis, calibrated_mod

        gc.collect()
        print("------------------------------\n")

    def save_prmts(self, name='params'):
        np.save(f"{name}.npy", self.params)

    def load_prmts(self, name='params'):
        self.params = np.load(f"{name}.npy", allow_pickle=True)

if __name__ == "__main__":

    print("\n     *** FRINGE FITTING CODE ***\n\n")
    # Path to the MeasurementSet (MS)
    data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

    refant = 0
    npol = [0,3]
    tintervals = 8
    nscaleg = 16
    LSM = False
    ms_data = LOAD_MS(data_MS, npol=npol, refant=refant)
    nt_loc = ms_data.nt // tintervals

    #### START MAIN LOOP
    
    for tint in range(tintervals):
        print(f"\n---> Time interval = {tint+1}/{tintervals}\n")
        tini = tint *nt_loc
        tend = ms_data.nt if tint == (tintervals-1) else (tint + 1) * nt_loc
        ms_data.flag_baselines(tini, tend)
        ms_data.load_data(tini, tend, corrcomb=2)

        Global_FF = FringeFitting(ms_data, tints=tintervals, nscale=nscaleg)
        Global_FF.FFT_init_global(ms_data, tint)
        print("TIME: ", tm.time()-INI_TIME)

        if LSM:
            Global_FF.LSM_FF(ms_data, tint, maxiter=1)
            print("TIME: ", tm.time()-INI_TIME)

        Global_FF.calibrate(tini, tend, ms_data, tint, model=True)

        print("TIME: ", tm.time()-INI_TIME)

    # END MAIN LOOP

    baseline_test = np.zeros([ms_data.nant, ms_data.nant])

    for ii in range(ms_data.nant):
        for jj in range(ii+1, ms_data.nant):
            blij = (jj*(jj-1))//2 + ii
            baseline_test[ii, jj] = ms_data.baseline_ok[blij]
            baseline_test[jj, ii] = ms_data.baseline_ok[blij]

    plt.figure(figsize=(8, 6))
    plt.imshow(baseline_test, origin='lower')
    plt.xlabel("Antenna 2")
    plt.ylabel("Antenna 1")
    plt.colorbar(label='% Flagged Data')
    plt.grid(False)
    plt.show()
 
    plt.figure(figsize=(8, 6))
    # for tint in range(Global_FF.tints):
    plt.plot(Global_FF.snr_, ".")
    plt.xlabel("Antenna 2")
    plt.ylabel("SNR_AIPS")
    plt.grid(False)

    plt.figure(figsize=(8, 6))
    # for tint in range(Global_FF.tints):
    plt.plot(Global_FF.snr1_, ".")
    plt.xlabel("Antenna 2")
    plt.ylabel("SNR_MEAN")
    plt.grid(False)

    plt.figure(figsize=(8, 6))
    # for tint in range(Global_FF.tints):
    plt.plot(Global_FF.sumw_, ".")
    plt.xlabel("Antenna 2")
    plt.ylabel("SUMW")
    plt.grid(False)
    plt.show()

    ms_data.close()

    print('\n           *** END ***\n')