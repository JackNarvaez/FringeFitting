"""
Calibration Testing Script
==========================

Provides a sandbox for testing and validating calibration method 
implementations, including effects such as dispersive delay.
"""

import numpy as np
import gc
import time as tm
import emcee
import corner
import matplotlib.pyplot as plt
from casacore.tables    import table
from scipy.fft          import fftshift, fft2, fftfreq
from scipy.optimize     import least_squares, differential_evolution
from scipy              import stats as st
from ..GlobalFF         import wrap_phase, snr_aips, SNR, baseline_ij, plot_phase
from scipy.optimize     import curve_fit

INI_TIME    = tm.time()
FLT_EPSILON = np.finfo(float).eps
M_PI    = np.pi
M_PI2   = 2*np.pi
M_PIP2  = np.pi/2

def log_to_file(filepath: str, value, skipline):
    """Append value to a file."""
    with open(filepath, "a", encoding="utf-8") as f:
        if skipline:
            f.write(f"{value}\n")
        else:
            f.write(f"{value}\t")

def gaussian(x, A, mu, sigma, C):
    """Gaussian + constant background"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

def disp_function(ms, kdisp):
    """Fitting function for dispersive delay"""
    return kdisp*(1/ms.freq + (ms.freq - ms.fmin - ms.fmax)/(ms.fmin*ms.fmax))

class LOAD_MS:
    def __init__(
        self,
        ms_path,
        ants,
        npol=0,
        SpW=0,
        refant=0,
        tints=8
    ):
        self.ms_path = ms_path
        self.n_pol   = npol

        # Load and process time
        self.data  = table(self.ms_path, readonly=True)
        self.time  = np.unique(self.data.getcol("TIME"))
        self.t0    = self.time[0]
        self.time -= self.t0
        self.nt    = len(self.time)
        self.dt    = self.time[1]-self.time[0]
        self.tints = tints

        # Time Intervals
        DT = np.diff(self.time)
        tindex = [0]
        tindex.extend(np.where(DT > self.dt)[0] + 1)
        tindex.append(self.nt)

        Numdivs = len(tindex)-1

        if (Numdivs > 1) and (Numdivs < 3*self.tints):  ### To keep Number of intervals small.
            print("Variable time intervals")
            self.tints = Numdivs
            self.time_edgs = tindex
        else:
            print("Uniform time intervals")
            self.time_edgs = np.arange(0, self.nt, self.nt//self.tints)
            self.time_edgs[-1] = self.nt

        print(f"Total Time steps: {self.nt}")
        print(f"Time Intervals  : {self.tints}")
        print(f"Interval's Edges: {self.time_edgs}")

        self.use_weight_spectrum = False
        if "WEIGHT_SPECTRUM" in self.data.colnames():
            self.use_weight_spectrum = True

        # Load frequency info
        spectral_window = table(f"{ms_path}/SPECTRAL_WINDOW")
        self.nSpW = len(spectral_window)
        self.SpW  = SpW
        self.freq = spectral_window[0]['CHAN_FREQ']
        self.nf   = len(self.freq)
        self.df   = self.freq[1] - self.freq[0]
        self.Dfrq = self.freq - self.freq[0]
        self.fmin = np.min(self.freq)
        self.fmax = np.max(self.freq)
        spectral_window.close()

        if (self.SpW >= self.nSpW):
            print(f"SpW = {self.SpW} must be smaller than {self.nSpW}.")
            exit()

        # Load antenna info
        antennas     = table(f"{ms_path}/ANTENNA")
        self.selants = ants
        self.nameant = np.array(antennas.getcol("NAME"))[ants]
        self.posant  = np.array(antennas.getcol("POSITION"))[ants]
        self.disunit = antennas.getcoldesc("POSITION")['keywords']['QuantumUnits'][0]
        self.nant    = len(ants)
        self.selfbls = -1
        self.n_bls   = self.nant*(self.nant + self.selfbls)//2
        antennas.close()

        self.refant  = refant

        print(f"Selected antennas (nant={self.nant}): {self.selants}")
        print(f"\t -> names: {self.nameant}")
        print(f"Reference Antenna: {self.nameant[self.refant]}")

        self.baseline_ok = np.zeros(self.n_bls, dtype=bool)

    def close(self):
        self.data.close()
    
    def load_baseline(
        self,
        ant1,
        ant2,
        npol=0,
        Data=True,
        Wgt=True,
        Cal=False,
        Model=False
    ):
        if (ant1 > ant2):
            print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
            exit()
        elif (ant1==ant2):
            if self.selfbls < 0:
                print(f"Error: Ant1: {ant1} must be different to Ant2: {ant2}")
                exit()
        t1   = self.data.query(f'DATA_DESC_ID == {self.SpW} AND ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')
        flg  = t1.getcol('FLAG')[:, :, npol]

        results = []
        if Data:
            vis  = t1.getcol('DATA')[:, :, npol]
            vis[flg] = 0.0
            results.append(vis)
        if Wgt:
            if self.use_weight_spectrum:
                wgt = t1.getcol('WEIGHT_SPECTRUM')[:, :, npol]
            else:
                # Use WEIGHT and broadcast it across channels
                weights = t1.getcol('WEIGHT')[:, npol]
                wgt = weights[:, np.newaxis].repeat(self.nf, axis=1)
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

    def flag_baselines(
        self,
        tini,
        tend,
        flagged_threshold=0.9,
        uvrange=7e4
    ):
        print("\n ------- FLAGGING DATA ------- ")
        print(f"flagged_threshold = {flagged_threshold:.2f}")
        print(f"uvrange_threshold = {uvrange:.2f}")
        self.flg_th = flagged_threshold
        self.uvw_th = uvrange
        ntot = self.nf*(tend-tini)
        if not isinstance(self.n_pol, int):
            ntot *= len(self.n_pol)
        t_offset = tini
        t_limit  = (tend-tini)

        for iant in range(self.nant):
            for jant in range(iant+1, self.nant):
                Tab = self.data.query(query=f"DATA_DESC_ID == {self.SpW} AND ANTENNA1 == {self.selants[iant]} AND ANTENNA2 == {self.selants[jant]}", offset=t_offset, limit=t_limit, columns="FLAG")
                flg = Tab.getcol('FLAG')[:, :, self.n_pol]
                Tab.close()
                temp_sum = np.sum(flg)
                bij   = baseline_ij(iant, jant, self.selfbls)
                cond1 = np.linalg.norm(self.posant[jant] - self.posant[iant]) > self.uvw_th
                cond2 = (self.selants[iant] == self.refant) or (self.selants[jant] == self.refant)
                if not (cond1 or cond2):
                    ok = False
                else:
                    ok = temp_sum/ntot < self.flg_th
                self.baseline_ok[bij] = ok
        print(f"Active Baselines: {np.sum(self.baseline_ok)} / {self.n_bls}")
        print("------------------------------\n")

    def load_data(self, corrcomb=0, Artificial=False, Model=True):
        self.vis = np.zeros([self.nt*self.n_bls, self.nf], dtype="complex")
        self.wgt = np.zeros([self.nt*self.n_bls, self.nf], dtype="float")
        self.calibrated_mod = np.zeros(self.vis.shape, dtype="float")
        self.calibrated_vis = np.zeros_like(self.vis)

        print("\n ------- LOADING DATA ------- ")
        cols = "DATA,FLAG"
        if Model:
            print("Using column: MODEL_DATA")
            cols = cols + ",MODEL_DATA"
        
        if self.use_weight_spectrum:
            print("Using column: WEIGHT_SPECTRUM")
            cols = cols + ",WEIGHT_SPECTRUM"
        else:
            cols = cols + ",WEIGHT"
            print("Using column: WEIGHT")

        if Artificial:
            self.wgt += 1.0
            self.baseline_ok[:] = True
            prms = [np.pi/2, 1, 100]
            for iant in range(self.nant):
                for jant in range(iant+1, self.nant):
                    bl_ij = baseline_ij(iant, jant, self.selfbls)
                    self.vis[bl_ij*self.nt:(bl_ij+1)*self.nt] = np.exp(1j*(prms[0]+M_PI2*np.add.outer(prms[1]/1e3*(self.time-self.time[0]), prms[2]/1e9*self.Dfrq) + M_PI2*0.1*np.random.randn(self.nt, self.nf)))

        else:
            for iant in range(self.nant):
                for jant in range(iant+1, self.nant):
                    bl_ij = baseline_ij(iant, jant, self.selfbls)
                    Tab = self.data.query(query=f"DATA_DESC_ID == {self.SpW} AND ANTENNA1 == {self.selants[iant]} AND ANTENNA2 == {self.selants[jant]}", columns=cols)
                    temp_vis = Tab.getcol('DATA')[:, :, self.n_pol]
                    temp_flg = Tab.getcol('FLAG')[:, :, self.n_pol]
                    if Model:
                        model_data =  Tab.getcol('MODEL_DATA')[:, :, self.n_pol]
                    # Check for WEIGHT_SPECTRUM
                    if self.use_weight_spectrum:
                        temp_wgt = Tab.getcol('WEIGHT_SPECTRUM')[:, :, self.n_pol]
                    else:
                        # Use WEIGHT and broadcast it across channels
                        weights = Tab.getcol('WEIGHT')[:, self.n_pol]
                        temp_wgt = weights[:, np.newaxis, :].repeat(self.nf, axis=1)
                    Tab.close()

                    # Combine correlations
                    if corrcomb==0:
                        print("Polarization: ", self.n_pol)
                    elif corrcomb==1:
                        print("Combining correlations: weighted-average")
                        temp_flg = np.all(temp_flg, axis=2)
                        temp_vis = np.sum(temp_vis*temp_wgt, axis=2)/np.sum(temp_wgt, axis=2)
                        temp_wgt = np.mean(temp_wgt, axis=2)
                        if Model:
                            model_data = np.sum(model_data*temp_wgt, axis=2)/np.sum(temp_wgt, axis=2)
                    elif corrcomb==2:
                        print("Combining correlations: Stokes I")
                        temp_flg = np.all(temp_flg, axis=2)
                        temp_vis = 0.5*np.sum(temp_vis, axis=2)
                        temp_wgt = 0.5*np.sum(temp_wgt, axis=2)
                        if Model:
                            model_data = 0.5*np.sum(model_data, axis=2)
                    else:
                        print(f"corrcomb={corrcomb} is not defined.")
                        exit()

                    temp_vis[temp_flg] = 0.0
                    temp_flg[:, :3] = True
                    temp_flg[:, self.nf-3:] = True
                    temp_wgt[temp_flg] = 0.0

                    if Model:
                        self.vis[bl_ij*self.nt:(bl_ij+1)*self.nt] = np.exp(1j*(np.angle(temp_vis) - np.angle(model_data)))
                    else:
                        self.vis[bl_ij*self.nt:(bl_ij+1)*self.nt] = np.exp(1j*np.angle(temp_vis))
                    self.wgt[bl_ij*self.nt:(bl_ij+1)*self.nt] = temp_wgt
            print("------------------------------\n")

            del temp_vis, temp_flg, temp_wgt

            gc.collect()
    
    def plot_phase_local(
        self,
        ant1,
        ant2,
        tini,
        tend,
        vis_cal=None,
        vis_wgt=None,
        showw=False
    ):

        baseline = baseline_ij(ant1, ant2, self.selfbls)
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
        plot_phase(phase_global, wgt_global, time_loc/3600, self.freq,\
                   nbins_time=len(time_loc), nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle, showw=showw)

class FringeFitting:
    def __init__(self, ms, nscale=8, tunit=1e3, funit=1e9, kunit=1e6):

        self.nscale = nscale
        self.tunit  = tunit
        self.funit  = funit
        self.kunit  = kunit
        self.params = np.zeros((ms.nant, 4))
        self.sumw_  = np.zeros((ms.nant))
        self.snr_   = np.zeros((ms.nant))
        self.snr1_  = np.zeros((ms.nant))

        # Constrains
        self.min_delay = self.funit * 0.5/(ms.freq[-1]-ms.freq[0])
        self.max_delay = self.funit * 0.5/ms.df
        self.min_rate  = self.tunit * 0.5/(ms.time[ms.time_edgs[1]] - ms.time[ms.time_edgs[0]])
        self.max_rate  = self.tunit * 0.5/ms.dt
        self.min_disp  = -5000
        self.max_disp  = 5000

    def FFT_init_ij(self, ms, ant, nt_loc, fringe_rate, delay, ti):
        ant1, ant2 = (ant, ms.refant) if ms.refant > ant else (ms.refant, ant)
        sgn = -1 if ant < ms.refant else 1

        bl_ij = baseline_ij(ant1, ant2, ms.selfbls)

        if ms.baseline_ok[bl_ij]:
            tt1 = bl_ij*ms.nt+ms.time_edgs[ti]
            tt2 = bl_ij*ms.nt+ms.time_edgs[ti+1]

            # 2D FFT
            F_sub = fftshift(fft2(ms.wgt[tt1:tt2]*ms.vis[tt1:tt2], s=[self.nscale*nt_loc, self.nscale*ms.nf]))
            amp_loc_sh = F_sub.shape

            #IMPLEMENT CONSTRAINS
            F_sub[amp_loc_sh[0]//2, :] = 0.0
            F_sub[:, amp_loc_sh[1]//2] = 0.0

            ampli_loc = np.abs(F_sub)

            # Find peak
            r_idx, tau_idx = np.unravel_index(np.argmax(ampli_loc), amp_loc_sh)

            self.params[ant, 0] = sgn*np.angle(F_sub[r_idx, tau_idx])

            rate_ini  = sgn*fringe_rate[r_idx] * self.tunit
            delay_ini = sgn*delay[tau_idx] * self.funit

            self.sumw_[ant] = np.sum(ms.wgt[tt1:tt2])
            sumww_ = np.sum(ms.wgt[tt1:tt2]**2)
            xcount_ = np.sum(ms.wgt[tt1:tt2]>0)
            self.snr_[ant] = snr_aips(np.abs(F_sub[r_idx, tau_idx]), self.sumw_[ant], sumww_, xcount_)
            self.snr1_[ant] = SNR(ampli_loc)

            ########################################
            # Calculate sigma for delay peak
            slice_tau = np.abs(F_sub[r_idx, :])
            x = np.arange(amp_loc_sh[1])

            # initial guesses for fit
            A0 = slice_tau.max() - slice_tau.min()
            mu0 = tau_idx
            sigma0 = 5.0  # some reasonable width guess
            C0 = slice_tau.min()

            # fit Gaussian
            popt, _ = curve_fit(gaussian, x, slice_tau, p0=[A0, mu0, sigma0, C0])

            sigma_tau = popt[2]

            print("Sigma: ", sigma_tau)
            # log_to_file("sigma_disp4.txt", sigma_tau, False)

            # plt.figure()
            # plt.plot(ms.freq/1e6, np.mean(ms.vis[tt1:tt2], axis=0), ".", ms=1)
            # # plt.plot(x, gaussian(x, popt[0], popt[1], popt[2], popt[3]), lw=1)
            # plt.show()
            ########################################

        else:
            rate_ini  = sgn*self.min_rate
            delay_ini = sgn*self.min_delay

        self.params[ant, 1] = rate_ini   
        self.params[ant, 2] = delay_ini
        self.params[ant, 3] = 0.0

    def FFT_init_global(self, ms, tint):
        print("\n ------- INITIAL GUESS ------- ")
        print("Constrains")
        print(f"RATE: min: {self.min_rate:.2f} mHz \t max: {self.max_rate:.2f} mHz")
        print(f"DELAY: min: {self.min_delay:.2f} ns \t max: {self.max_delay:.2f} ns")
        print("------------------------------\n")
        nt_loc      =  ms.time_edgs[tint+1]-ms.time_edgs[tint]
        fringe_rate = fftshift(fftfreq(self.nscale * nt_loc, ms.dt))  # Hz
        delay       = fftshift(fftfreq(self.nscale * ms.nf, ms.df))
        ddf = 10
        for iant in range(ms.nant):
            if (ms.refant == iant):
                continue
            self.FFT_init_ij(ms, iant, nt_loc, fringe_rate, delay, tint)
        
        time_new = ms.time[ms.time_edgs[tint]:ms.time_edgs[tint+1]]

        for iant in range(ms.nant):
            if (ms.refant == iant):
                continue
            # phaMOD = self.params[iant, 0]+M_PI2*np.add.outer(self.params[iant, 1]/self.tunit*time_new, self.params[iant, 2]/self.funit*ms.Dfrq [ms.nf//2 - ddf:ms.nf//2 + ddf])
            # intbl = baseline_ij(iant, ms.refant, ms.selfbls)*ms.nt
            # phaVIS = np.angle(ms.vis[intbl + ms.time_edgs[tint]:intbl + ms.time_edgs[tint+1], ms.nf//2 - ddf:ms.nf//2 + ddf])
            # Sumdis = (phaVIS - phaMOD)/(M_PI2*self.kunit*(1/ms.freq[ms.nf//2 - ddf:ms.nf//2 + ddf] + (ms.freq[ms.nf//2 - ddf:ms.nf//2 + ddf] - ms.fmin - ms.fmax)/(ms.fmin*ms.fmax)) + FLT_EPSILON)
            # self.params[iant, 3] = np.sum(Sumdis)/(2*ddf*nt_loc)
            print(f"Baseline: {iant} \t phase: {self.params[iant, 0]:.2f} rad \t rate: {self.params[iant, 1]:.2f}\t delay: {self.params[iant, 2]:.2f} ns \t disp: {self.params[iant, 3] :.2f} \t snr_mean {self.snr1_[iant]:.2f}\t snr_aips: {self.snr_[iant]:.2f}")

        print("Initial Guess set successfully\n")

    def S3(self, vis_loc, wgt_loc, Dt, ms, prms):
        phi0  = wrap_phase(prms[0])
        rDt   = (prms[1]/self.tunit)*Dt
        tauDf = (prms[2]/self.funit)*ms.Dfrq
        kdisp = (prms[3]*self.kunit)*(1/ms.freq + (ms.freq - ms.fmin - ms.fmax)/(ms.fmin*ms.fmax))
        fCorr = tauDf + kdisp

        Eijk = np.exp(1j*(phi0 + M_PI2*np.add.outer(rDt,fCorr)))
        S2_t = np.abs(vis_loc-Eijk)**2
        S    = np.sum(wgt_loc*S2_t)
        return S

    def GlobalCost(self, params, ms, tint, time_n):
        nprms = 4
        x0 = np.zeros(nprms*ms.n_bls)
        for iant in range(ms.nant):
            if iant < ms.refant:
                prmsi = params[nprms*iant:nprms*(iant+1)]
            elif iant > ms.refant:
                prmsi = params[nprms*(iant-1):nprms*iant]
            else:
                prmsi = np.zeros(nprms)
            for jant in range(iant+1, ms.nant):
                if jant < ms.refant:
                    prmsj = params[nprms*jant:nprms*(jant+1)]
                elif jant > ms.refant:
                    prmsj = params[nprms*(jant-1):nprms*jant]
                else:
                    prmsj = np.zeros(nprms)
                bl_ij = baseline_ij(iant, jant, ms.selfbls)
                if ~ms.baseline_ok[bl_ij]:
                    continue
                x0[nprms*bl_ij:nprms*(bl_ij+1)] = prmsj - prmsi

        S = 0.0
        for bl_ij in range(ms.n_bls):
            if not ms.baseline_ok[bl_ij]:
                continue
            tt1 = bl_ij*ms_data.nt + ms_data.time_edgs[tint]
            tt2 = bl_ij*ms_data.nt + ms_data.time_edgs[tint+1]
            vis_loc = ms.vis[tt1:tt2]
            wgt_loc = ms.wgt[tt1:tt2]

            prm_loc = x0[nprms*bl_ij:nprms*(bl_ij+1)]
            S += self.S3(vis_loc, wgt_loc, time_n, ms, prm_loc)
        return S
    
    def GlobalCost_disp(self, disp_only, ms, tint, time_n):
        nprms = 4
        x0 = np.zeros(nprms*ms.n_bls)

        # full parameter set (with first 3 fixed)
        full_params = self.params.flatten()

        for iant in range(ms.nant):
            if iant == ms.refant:
                continue
            full_params[nprms*iant + 3] = disp_only[iant if iant < ms.refant else iant-1]

        # now build baseline differences
        for iant in range(ms.nant):
            prmsi = full_params[nprms*iant:nprms*(iant+1)]
            for jant in range(iant+1, ms.nant):
                prmsj = full_params[nprms*jant:nprms*(jant+1)]
                bl_ij = baseline_ij(iant, jant, ms.selfbls)
                if ~ms.baseline_ok[bl_ij]:
                    continue
                x0[nprms*bl_ij:nprms*(bl_ij+1)] = prmsj - prmsi

        # cost evaluation
        S = 0.0
        for bl_ij in range(ms.n_bls):
            if not ms.baseline_ok[bl_ij]:
                continue
            tt1 = bl_ij*ms_data.nt + ms_data.time_edgs[tint]
            tt2 = bl_ij*ms_data.nt + ms_data.time_edgs[tint+1]
            vis_loc = ms.vis[tt1:tt2]
            wgt_loc = ms.wgt[tt1:tt2]

            prm_loc = x0[nprms*bl_ij:nprms*(bl_ij+1)]
            S += self.S3(vis_loc, wgt_loc, time_n, ms, prm_loc)
        return S

    def LSM_FF(self, ms, tint, maxiter=50, lsm_="local"):
        print("\n ------- GLOBAL LSM ------- ")

        nprms = 4
        
        low_bounds = []
        upp_bounds = []
        for _ in range(ms.nant-1):
            low_bounds.extend([-M_PI,-self.max_rate, -self.max_delay, -self.max_disp])  # phi0, r, tau
            upp_bounds.extend([ M_PI, self.max_rate,  self.max_delay,  self.max_disp])  # phi0, r, tau

        x0   = self.params.flatten()
        x0   = np.delete(x0, slice(nprms*ms.refant, nprms*(ms.refant+1)))

        tini = ms.time_edgs[tint]
        tend = ms.time_edgs[tint+1]
        time_new = ms.time[tini:tend]-ms.time[tini]
        Err0 = self.GlobalCost(x0, ms, tint, time_new)
        if lsm_ == "local":
            bounds  = np.array([low_bounds, upp_bounds]).T
            result  = least_squares(self.GlobalCost, x0, bounds=(low_bounds, upp_bounds), args=(ms, tint, time_new), max_nfev=maxiter, verbose=1)
            end_res = result.fun[0]
        elif lsm_ == "global":
            bounds  = np.array([low_bounds, upp_bounds]).T
            result  = differential_evolution(self.GlobalCost, x0=x0, args=(ms, tint, time_new), bounds=bounds, maxiter=maxiter)
            end_res = result.fun
        elif lsm_ == "disp":
            low_bounds = -self.max_disp*np.ones(ms.nant-1)
            upp_bounds = self.max_disp*np.ones(ms.nant-1)
            disp_init  = np.delete(self.params[:, 3], ms.refant)
            result  = least_squares(self.GlobalCost_disp, disp_init, bounds=(low_bounds, upp_bounds), args=(ms, tint, time_new), max_nfev=maxiter, verbose=1)
            end_res = result.fun[0]
        else:
            print(f"lsm: {lsm_} not defined. Must be either 'local' or 'global'")
            exit()
        if lsm_== "disp":
            xn = x0
            xn[3::nprms] = result.x
        else:
            xn = result.x
        Errn = self.GlobalCost(xn, ms, tint, time_new)
        print(f"residual: {end_res:.2f}")
        print(f"residual: {Err0:.2f}")
        print(f"Relative residual change: {(Err0 - Errn)/Err0:.3f}")
        x0_full = np.insert(xn, nprms*ms.refant, np.zeros(nprms))

        self.params = x0_full.reshape(ms.nant, nprms)

        for iant in range(ms.nant):
            if (ms.refant == iant):
                continue
            print(f"Baseline: {iant} \t phase: {self.params[iant, 0]:.2f} rad \t rate: {self.params[iant, 1]:.2f}\t delay: {self.params[iant, 2]:.2f} ns \t disp: {self.params[iant, 3] :.2f}")


        print(f"LSM is finished. Max. iteration is {maxiter}")
        print("------------------------------\n")

    def log_prior(self, params, ms):
        for i in range(ms.nant-1):
            phi, r, tau, kdisp = params[4*i:4*i+4]
            if not ((-M_PI <= phi <= M_PI) and (-self.max_rate <= r <= self.max_rate) and (-self.max_delay <= tau <= self.max_delay) and (-self.max_disp <= kdisp <= self.max_disp)):
                return -np.inf
        return 0.0

    def log_probability(self, params, ms, tint, time_n):
        lp = self.log_prior(params, ms)
        S3 = 0.5*self.GlobalCost(params, ms, tint, time_n)
        if not np.isfinite(lp):
            return -np.inf
        return lp - S3
    
    def run_emcee(self, ms, tint, nwalkers=32, steps=1000, verbose=False):
        print("\n ------- GLOBAL MCMC (emcee) ------- ")

        global errrorrr
        nprms = 4
        ndim = nprms*(ms.nant-1)
        x0   = self.params.flatten()
        x0   = np.delete(x0, slice(nprms*ms.refant, nprms*(ms.refant+1)))

        tini = ms.time_edgs[tint]
        tend = ms.time_edgs[tint+1]
        time_new = ms.time[tini:tend]-ms.time[tini]
        Err0 = self.GlobalCost(x0, ms, tint, time_new)

        low_bounds = []
        upp_bounds = []
        for _ in range(ms.nant-1):
            low_bounds.extend([-M_PI,-self.max_rate, -self.max_delay, -self.max_disp])  # phi0, r, tau
            upp_bounds.extend([ M_PI, self.max_rate,  self.max_delay,  self.max_disp]) 

        p0 = np.random.uniform(low_bounds,
                       upp_bounds,
                       size=(nwalkers, ndim))
        # p0   = x0 + 1e-1 * (np.random.randn(nwalkers, ndim)-0.5)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,
                                        args=(ms, tint, time_new))

        sampler.run_mcmc(p0, steps, progress=True)

        samples = sampler.get_chain(discard=100, thin=2, flat=True)

        costs = np.array([-np.log(self.GlobalCost(s, ms, tint, time_new)/Err0) for s in samples])

        if verbose:
            # fig, axes = plt.subplots(4, figsize=(10, 10), sharex=True)
            # samples1 = sampler.get_chain(discard=100, thin=2, flat=False)

            # labels = ["phi", "r", "tau"]
            # for i in range(ndim):
            #     ax = axes[i]
            #     ax.plot(samples1[:, :, i], "w", lw=0.5, alpha=0.3)
            #     ax.set_xlim(0, len(samples1))
            #     ax.set_ylabel(labels[i])
            #     ax.yaxis.set_label_coords(-0.1, 0.5)

            # ax = axes[3]
            # ax.plot(costs.reshape(len(samples1), nwalkers), "w", lw=0.5, alpha=0.3)
            # ax.set_xlim(0, len(samples1))
            # ax.set_ylim(0)
            # ax.set_ylabel("cost")
            # ax.yaxis.set_label_coords(-0.1, 0.5)

            # axes[-1].set_xlabel("step number");

            fig = corner.corner(
                samples,
                bins=100,
                # labels=labels,
                # quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 10},
                plot_contours=False,            # disables level curves
                plot_datapoints=True,           # shows raw data points
                lines_kwargs={"lw": 1},        # thin percentile lines
                hist_kwargs={"linewidth": 1},
            )
            plt.show()

        # Find the index of the sample with the minimum GlobalCost
        # best_fit = np.median(samples, axis=0)
        # best_fit = st.mode(samples, axis=0)
        best_idx = np.argmax(costs)

        # Take the corresponding sample
        best_fit = samples[best_idx]

        Errn = self.GlobalCost(best_fit, ms, tint, time_new)
        errrorrr.append((Err0 - Errn)/Err0)
        print(f"Relative residual change: {(Err0 - Errn)/Err0:.3f}")

        x0_full = np.insert(best_fit, nprms*ms.refant, np.zeros(nprms))

        self.params = x0_full.reshape(ms.nant, nprms)

        for iant in range(ms.nant):
            if (ms.refant == iant):
                continue
            print(f"Baseline: {iant} \t phase: {self.params[iant, 0]:.2f} rad \t rate: {self.params[iant, 1]:.2f}\t delay: {self.params[iant, 2]:.2f} ns \t disp: {self.params[iant, 3] :.2f}")
            # log_to_file("sigma_disp4.txt", self.params[iant, 3], True)

        print("Optimization complete (MCMC).")
        print("----------------------------------------------\n")

    def calibrate(self, tint, tini, tend, ms):
        """ Save calibrated data in MS table"""
        tloc = tend-tini
        
        print("\n ------- DATA CALIBRATION ------- ")

        print("Start calibration...")
        for iant in range(ms.nant):
            prmsi = self.params[iant]
            for jant in range(iant+1, ms.nant):
                bl_ij = baseline_ij(iant, jant, ms.selfbls)
                tt1 = bl_ij*ms.nt+ms.time_edgs[tint]
                if ~ms.baseline_ok[bl_ij]:
                    ms.calibrated_vis[tt1:tt1+tloc] = ms.vis[tt1:tt1+tloc]
                    continue
                prms = self.params[jant] - prmsi
                for tt in range(tloc):
                    phase_mod = prms[0]+M_PI2*(prms[1]/self.tunit*(ms.time[tini+tt]-ms.time[tini])
                                         + prms[2]/self.funit*ms.Dfrq
                                         + prms[3]*self.kunit*(1/ms.freq + (ms.freq - ms.fmin - ms.fmax)/(ms.fmin*ms.fmax)))
                    G_ff_inv = np.exp(-1j*phase_mod)
                    ms.calibrated_vis[tt1+tt] = ms.vis[tt1+tt]*G_ff_inv
                    ms.calibrated_mod[tt1+tt] = np.angle(G_ff_inv)

                # plt.figure()
                ydata = np.mean(ms.calibrated_vis[tt1:tt1+tloc], axis=0)
                # plt.plot(ms.freq/1e6, ydata, ".", ms=3)

                popt, _ = curve_fit(disp_function, ms, ydata, p0=[100])

                # plt.plot(ms.freq/1e6, disp_function(ms.freq, popt[0]), "--", c="y")
                # plt.show()

                for tt in range(tloc):
                    phase_mod = popt[0]*(1/ms.freq + (ms.freq - ms.fmin - ms.fmax)/(ms.fmin*ms.fmax))
                    G_ff_inv = np.exp(-1j*phase_mod)
                    ms.calibrated_vis[tt1+tt] *= G_ff_inv

                # plt.figure()
                # plt.plot(M_PI2*(prms[3]*self.kunit*(1/ms.freq + (ms.freq - ms.fmin - ms.fmax)/(ms.fmin*ms.fmax))), ".")
                # plt.plot(M_PI2*(prms[2]/self.funit*ms.Dfrq), ".")
                # plt.show()
        
        print("Calibration complete")
        print("------------------------------\n")

if __name__ == "__main__":

    print("\n     *** FRINGE FITTING CODE ***\n\n")
    # Path to the MeasurementSet (MS)
    # data_MS  = "../Data/n24l2.ms"

    # npol        = [0, 3]
    # nSpW        = 1
    # tintervals  = 8
    # nscaleg     = 16
    # LSM         = False
    # lsm_        = "emcee"
    # selfbls     = True
    # vlbi        = "evn"
    # uvrange     = 0
    # real_refant = 2
    # selectants  = [real_refant, 5, 8]
    # refant      = 0

    data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

    npol        = [0, 3]
    nSpW        = 0
    tintervals  = 80
    nscaleg     = 32
    LSM         = False
    lsm_        = "emcee"
    selfbls     = False
    vlbi        = "lofar"
    uvrange     = 0
    real_refant = 26
    selectants  = [18, real_refant]
    refant      = 1

    ms_data     = LOAD_MS(data_MS, ants=selectants, npol=npol, SpW=nSpW, refant=refant, tints=tintervals)
    ms_data.load_data(corrcomb=2, Model=False)

    # bl1 = baseline_ij(0, 1, ms_data.selfbls)
    # bl2 = baseline_ij(0, 2, ms_data.selfbls)
    # bl3 = baseline_ij(1, 2, ms_data.selfbls)

    # totvis = wrap_phase(np.angle(ms_data.vis[bl1*ms_data.nt:(bl1+1)*ms_data.nt]) - np.angle(ms_data.vis[bl2*ms_data.nt:(bl2+1)*ms_data.nt]))
    # totvis = wrap_phase(totvis + np.angle(ms_data.vis[bl3*ms_data.nt:(bl3+1)*ms_data.nt]))

    # plt.figure()
    # plt.imshow(np.angle(ms_data.vis).T, vmax=np.pi, vmin=-np.pi, cmap="twilight", origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.imshow(ms_data.wgt[bl1*ms_data.nt:(bl1+1)*ms_data.nt].T, vmin=0, cmap="magma", origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.show()

    #### START MAIN LOOP
    for tint in range(ms_data.tints):
        print(f"\n---> Time interval = {tint+1}/{ms_data.tints}")
        tini = ms_data.time_edgs[tint]
        tend = ms_data.time_edgs[tint+1]
        print(f"Time steps: {tend-tini}\n")
        ms_data.flag_baselines(tini, tend, uvrange = uvrange)

        Global_FF = FringeFitting(ms_data, nscale=nscaleg)
        Global_FF.FFT_init_global(ms_data, tint)
        x0_par   = Global_FF.params.flatten()
        x0_par   = np.delete(x0_par, slice(3*ms_data.refant, 3*(ms_data.refant+1)))
        print(f"Error: {Global_FF.GlobalCost(x0_par, ms_data, tint, ms_data.time[tini:tend]-ms_data.time[tini])}")
        print("TIME: ", tm.time()-INI_TIME)

        if LSM:
            if lsm_ == "emcee":
                Global_FF.run_emcee(ms_data, tint, nwalkers=32*len(selectants), steps=2000, verbose=False)
            else:
                Global_FF.LSM_FF(ms_data, tint, maxiter=10000, lsm_=lsm_)
            print("TIME: ", tm.time()-INI_TIME)

        Global_FF.calibrate(tint, tini, tend, ms_data)

    # totcalvis = wrap_phase(np.angle(ms_data.calibrated_vis[bl1*ms_data.nt:(bl1+1)*ms_data.nt]) - np.angle(ms_data.calibrated_vis[bl2*ms_data.nt:(bl2+1)*ms_data.nt]))
    # totcalvis = wrap_phase(totcalvis + np.angle(ms_data.calibrated_vis[bl3*ms_data.nt:(bl3+1)*ms_data.nt]))

    # plt.figure()
    # plt.imshow(totcalvis.T, vmax=np.pi, vmin=-np.pi, cmap="twilight", origin="lower", aspect="auto", extent=[ms_data.time[0]/3600, ms_data.time[-1]/3600, ms_data.freq[0]/1e6, ms_data.freq[-1]/1e6])
    # plt.xlabel("Time (h)")
    # plt.ylabel("Freq. (MHz)")
    # plt.colorbar()
    # plt.show()

    for iant in range(ms_data.nant):
        for jant in range(iant+1, ms_data.nant):
            blij = baseline_ij(iant, jant, ms_data.selfbls)
            ms_data.plot_phase_local(iant, jant, 0, ms_data.nt, ms_data.vis[blij*ms_data.nt:(blij+1)*ms_data.nt], vis_wgt=ms_data.wgt[blij*ms_data.nt:(blij+1)*ms_data.nt], showw=False)
            ms_data.plot_phase_local(iant, jant, 0, ms_data.nt, ms_data.calibrated_vis[blij*ms_data.nt:(blij+1)*ms_data.nt], vis_wgt=ms_data.wgt[blij*ms_data.nt:(blij+1)*ms_data.nt], showw=False)
            ms_data.plot_phase_local(iant, jant, 0, ms_data.nt, np.exp(1j*ms_data.calibrated_mod[blij*ms_data.nt:(blij+1)*ms_data.nt]), showw=True)

    ms_data.close()
    print('\n           *** END ***\n')