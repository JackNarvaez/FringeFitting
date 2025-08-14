import numpy as np
import gc
import time as tm
import matplotlib.pyplot as plt
from casacore.tables     import table, makearrcoldesc, maketabdesc
from matplotlib.gridspec import GridSpec
from scipy.fft           import fftshift, fft2, fftfreq
from scipy.optimize      import least_squares, differential_evolution
from matplotlib import rcParams

plt.style.use('dark_background')


rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.markersize": 7,
    "lines.linewidth": 4,
    "figure.figsize": (8, 8),
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": 'in',
    "ytick.direction": 'in'
})

INI_TIME    = tm.time()
FLT_EPSILON = np.finfo(float).eps
M_PI    = np.pi
M_PI2   = 2*np.pi
M_PIP2  = np.pi/2

def wrap_phase(phase):
    """Fast phase wrapping"""
    return ((phase + M_PI) % (M_PI2)) - M_PI

def snr_aips(peak, sumw_, sumww_, xcount_):
    """Estimate signal-to-noise ratio based on AIPS formula."""
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
    return np.max(ampl_FFT) / (np.mean(ampl_FFT) + FLT_EPSILON)

def baseline_ij(ant1, ant2, sgn):
    """ Compute baseline index """
    if (ant2 > ant1):
        return (ant2*(ant2+sgn))//2 + ant1
    elif (ant2 < ant1):
        return (ant1*(ant1+sgn))//2 + ant2
    else:
        if sgn>0:
            return (ant1*(ant1+sgn))//2 + ant2
        else:
            print(f"Error: Ant1: {ant1} must be different to Ant2: {ant2}")
            exit()

def plot_phase(phase, wgt_glb, time, freq, nbins_time, nbins_freq, nbins_phase = 50, Baselinetitle="", showw=False, savef=False):
        fig = plt.figure(figsize=(12, 7))
        fig.suptitle(f"{Baselinetitle}")
        gs  = GridSpec(3, 3, width_ratios=[8, 1, 1], height_ratios=[1, 1, 4],
                    wspace=0.05, hspace=0.05)

        ax_top = fig.add_subplot(gs[0, 0])
        ax_top.plot(time, np.std(phase, axis=1), ".", ms=1, color="forestgreen")
        ax_top.set_xlim(time[0], time[-1])
        ax_top.set_ylabel("std")
        ax_top.tick_params(
            axis='y',
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
            direction='in'
        )
        ax_top.set_ylim(0, M_PI)
        ax_top.set_xticklabels([])
        ax_top.set_yticks([0, np.pi])
        ax_top.set_yticklabels([0, r"$\pi$"])

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
            cmap='plasma', cmin=1
        )

        ax_top2.set_ylabel("phase")
        ax_top2.set_xlim(time[0], time[-1])
        ax_top2.set_ylim(-M_PI, M_PI)
        ax_top2.set_xticklabels([])
        ax_top2.set_yticks([-np.pi, 0, np.pi])
        ax_top2.set_yticklabels([r"$-\pi$", 0, r"$\pi$"])

        ax_right = fig.add_subplot(gs[2, 2])
        ax_right.plot(np.std(phase,axis=0), freq, ".", ms=5, color='green')
        ax_right.set_ylim(freq[0], freq[-1])
        ax_right.set_xlabel("std")

        ax_right.tick_params(
            axis='x',
            bottom=True,
            top=True,
            labelbottom=False,
            labeltop=True,
            direction='in'
        )
        ax_right.set_xlim(0, M_PI)
        ax_right.set_yticklabels([])
        ax_right.set_xticks([0, np.pi])
        ax_right.set_xticklabels([0, r"$\pi$"])

        ax_right2 = fig.add_subplot(gs[2, 1])
        freq_rep    = np.repeat(freq[np.newaxis, :], phase.shape[0], axis=0).flatten()
        ax_right2.hist2d(
            phase.flatten(), freq_rep,
            weights = wgt,
            bins=[nbins_phase, nbins_freq],
            range=[[-M_PI, M_PI], [freq[0], freq[-1]]],
            cmap='plasma', cmin=1
        )

        ax_right2.set_xlabel("phase")
        ax_right2.set_ylim(freq[0], freq[-1])
        ax_right2.set_xlim(-M_PI, M_PI)
        ax_right2.set_yticklabels([])
        ax_right2.set_xticks([-np.pi, 0, np.pi])
        ax_right2.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])

        ax_main = fig.add_subplot(gs[2, 0])

        # ax_main.imshow(phase.T, aspect='auto', origin='lower', vmax=M_PI, vmin=-M_PI,
        #                 extent=[time[0], time[-1], freq[0]/1e6, freq[-1]/1e6],
        #                 cmap='twilight')

        dt = np.min(np.diff(time))
        time_full = np.arange(time.min(), time.max() + dt, dt)
        phase_full = np.full((time_full.size, phase.shape[1]), np.nan)

        idx_map = np.searchsorted(time_full, time)
        phase_full[idx_map, :] = phase

        T, F = np.meshgrid(time_full, freq)
        phase_masked = np.ma.masked_invalid(phase_full.T)

        cmap = plt.cm.twilight
        cmap.set_bad(color='black')

        pcm = ax_main.pcolormesh(T, F/1e6, phase_masked,
                         cmap=cmap, vmin=-M_PI, vmax=M_PI,
                         shading='auto')
        
        plt.xlabel("Time (h)")
        plt.ylabel("Frequency (MHz)")
        if savef:
            plt.savefig(Baselinetitle + ".png", dpi=500, transparent=True)
        if showw:
            plt.show()

class LOAD_MS:
    def __init__(self, ms_path, npol=0, SpW=0, refant=0, tints=8, selfbls=False, vlbi="lofar", model=False, bl_based=False):
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
        spectral_window.close()

        if (self.SpW >= self.nSpW):
            print(f"SpW = {self.SpW} must be smaller than {self.nSpW}.")
            exit()

        # Load antenna info
        antennas     = table(f"{ms_path}/ANTENNA")
        self.nameant = antennas.getcol("NAME")
        self.posant  = antennas.getcol("POSITION")
        self.disunit = antennas.getcoldesc("POSITION")['keywords']['QuantumUnits'][0]
        self.nant    = len(self.nameant)
        self.selfbls = 1 if selfbls else -1
        self.n_bls   = self.nant*(self.nant + self.selfbls)//2
        antennas.close()

        self.refant  = refant

        print(f"Reference Antenna: {self.nameant[self.refant]}")

        self.baseline_ok = np.zeros(self.n_bls, dtype=bool)

        # MS Format
        self.model  = model
        self.vlbi   = vlbi
        self.bl_based = bl_based
        self.order_bls = np.arange(0, self.n_bls)
        if self.vlbi=="lofar":
            print("VLBI MS Format: lofar")
        elif self.vlbi=="evn":
            print("VLBI MS Format: evn")
            Tab = self.data.query(query=f"DATA_DESC_ID == {self.SpW}", offset=0, limit=self.n_bls, columns="ANTENNA1,ANTENNA2")
            antenna1 = Tab.getcol('ANTENNA1')
            antenna2 = Tab.getcol('ANTENNA2')
            Tab.close()
            for irow in range(self.n_bls):
                self.order_bls[baseline_ij(antenna1[irow], antenna2[irow], self.selfbls)]=irow
        else:
            print(f"ERROR: unknown VLBI MS Format: {self.vlbi}. Must be 'lofar' or 'evn'")
            exit()

    def close(self):
        self.data.close()
    
    def load_baseline(self, ant1, ant2, npol=0, Data=True, Wgt=True, Cal=False, calmod=False, Model=False, fld=0):
        if (ant1 > ant2):
            print(f"Error: Ant1: {ant1} must be smaller than Ant2: {ant2}")
            exit()
        elif (ant1==ant2):
            if self.selfbls < 0:
                print(f"Error: Ant1: {ant1} must be different to Ant2: {ant2}")
                exit()
        t1   = self.data.query(f'DATA_DESC_ID == {self.SpW} AND FIELD_ID == {fld} AND ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}')
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
        if calmod:
            calmod = t1.getcol('CALIBRATED_MODEL')
            calmod[flg] = 0.0
            results.append(calmod)
        if Model:
            mod  = t1.getcol('MODEL_DATA')[:, :, npol]
            mod[flg] = 0.0
            results.append(mod)
        t1.close()

        return tuple(results)

    def flag_baselines(self, tini, tend, flagged_threshold = 0.6, uvrange = 2e4):
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
        cols = "FLAG,ANTENNA1,ANTENNA2"

        Tab = self.data.query(query=f"DATA_DESC_ID == {self.SpW}", offset=t_offset, limit=t_limit, columns=cols)

        NRows = Tab.nrows()
        if (NRows != (tend-tini)*self.n_bls):
            print(f"Nrows={NRows} must be equal to tloc*#Baselines={(tend-tini)*self.n_bls}")
            exit()

        flg = Tab.getcol('FLAG')[:, :, self.n_pol]

        if self.vlbi == "lofar":
            antenna1 = Tab.getcol('ANTENNA1')[:self.n_bls]
            antenna2 = Tab.getcol('ANTENNA2')[:self.n_bls]
        else:
            antenna1 = Tab.getcol('ANTENNA1')
            antenna2 = Tab.getcol('ANTENNA2')

        Tab.close()
        temp_sum = np.zeros(self.n_bls)

        if self.vlbi=="lofar":
            for irow in range(t_limit):
                bij = irow % self.n_bls;
                temp_sum[bij] += np.sum(flg[irow])
        else:
            for irow in range(t_limit):
                bij = baseline_ij(antenna1[irow], antenna2[irow], self.selfbls)
                temp_sum[bij] += np.sum(flg[irow])

        cond2 = False
        for iant in range(self.nant):
            for jant in range(iant+1, self.nant):
                bij   = baseline_ij(iant, jant, self.selfbls)
                cond1 = np.linalg.norm(self.posant[jant] - self.posant[iant]) > self.uvw_th
                if not self.bl_based:
                    cond2 = (iant == self.refant) or (jant == self.refant)
                if not (cond1 or cond2):
                    ok = False
                else:
                    ok = temp_sum[bij]/ntot < self.flg_th
                self.baseline_ok[bij] = ok
        self.baselinesON = np.sum(self.baseline_ok)
        print(f"Active Baselines: {self.baselinesON} / {self.n_bls}")
        print("------------------------------\n")

    def load_data(self, tini, tend, corrcomb=0):
        tloc = tend-tini
        ntot = self.nf*tloc
        t_offset = tini*self.n_bls
        t_limit  = tloc*self.n_bls

        print("\n ------- LOADING DATA ------- ")
        cols = "DATA,FLAG"
        if self.model:
            print("Using column: MODEL_DATA")
            cols = cols + ",MODEL_DATA"
        if self.use_weight_spectrum:
            print("Using column: WEIGHT_SPECTRUM")
            cols = cols + ",WEIGHT_SPECTRUM"
        else:
            cols = cols + ",WEIGHT"
            print("Using column: WEIGHT")
        
        if self.vlbi=="evn":
            cols = cols + ',ANTENNA1,ANTENNA2'

        Tab = self.data.query(f"DATA_DESC_ID == {self.SpW}", offset=t_offset, limit=t_limit, columns=cols)

        NRows = Tab.nrows()
        if (NRows != tloc*self.n_bls):
            print(f"Nrows={NRows} must be equal to tloc*#Baselines={tloc*self.n_bls}")
            exit()

        temp_vis = Tab.getcol('DATA')[:, :, self.n_pol]
        temp_flg = Tab.getcol('FLAG')[:, :, self.n_pol]
        if self.vlbi == "evn":
            antenna1 = Tab.getcol('ANTENNA1')
            antenna2 = Tab.getcol('ANTENNA2')

        # Check for WEIGHT_SPECTRUM
        if self.use_weight_spectrum:
            temp_wgt = Tab.getcol('WEIGHT_SPECTRUM')[:, :, self.n_pol]
        else:
            # Use WEIGHT and broadcast it across channels
            weights = Tab.getcol('WEIGHT')[:, self.n_pol]
            temp_wgt = weights[:, np.newaxis, :].repeat(self.nf, axis=1)

        if self.model:
            model_data =  Tab.getcol('MODEL_DATA')[:, :, self.n_pol]
        Tab.close()

        # Combine correlations
        if corrcomb==0:
            print("Polarization: ", self.n_pol)
        elif corrcomb==1:
            print("Combining correlations: weighted-average")
            temp_flg = np.all(temp_flg, axis=2)
            temp_vis = np.sum(temp_vis*temp_wgt, axis=2)/np.sum(temp_wgt, axis=2)
            if self.model:
                model_data = np.sum(model_data*temp_wgt, axis=2)/np.sum(temp_wgt, axis=2)
            temp_wgt = np.mean(temp_wgt, axis=2)
        elif corrcomb==2:
            print("Combining correlations: Stokes I")
            temp_flg = np.all(temp_flg, axis=2)
            temp_vis = 0.5*np.sum(temp_vis, axis=2)
            if self.model:
                model_data = 0.5*np.sum(model_data, axis=2)
            temp_wgt = 0.5*np.sum(temp_wgt, axis=2)
        else:
            print(f"corrcomb={corrcomb} is not defined.")
            exit()

        self.vis = np.zeros_like(temp_vis)
        self.wgt = np.zeros_like(temp_wgt)
        flg      = np.zeros_like(temp_flg)

        print("Ordering data. ")

        if self.vlbi=="lofar":
            for bl in range(self.n_bls):
                if self.baseline_ok[bl]:
                    for tt in range(tloc):
                        orig_row = tt * self.n_bls + bl
                        new_row  = bl * tloc + tt
                        if self.model:
                            self.vis[new_row] = np.exp(1j*(np.angle(temp_vis[orig_row])-np.angle(model_data[orig_row])))
                        else:
                            self.vis[new_row] = np.exp(1j*np.angle(temp_vis[orig_row]))
                        self.wgt[new_row] = temp_wgt[orig_row]
                        flg[new_row]      = temp_flg[orig_row]
        else:
            for nrow in range(NRows):
                if (antenna1[nrow] == antenna2[nrow]):
                    continue
                tt = nrow // self.n_bls
                bl = baseline_ij(antenna1[nrow], antenna2[nrow], self.selfbls)
                if self.baseline_ok[bl]:
                    new_row  = bl * tloc + tt
                    if self.model:
                        self.vis[new_row] = np.exp(1j*(np.angle(temp_vis[nrow])-np.angle(model_data[nrow])))
                    else:
                        self.vis[new_row] = np.exp(1j*np.angle(temp_vis[nrow]))
                    self.wgt[new_row] = temp_wgt[nrow]
                    flg[new_row]      = temp_flg[nrow]

        self.vis[flg] = 0.0
        flg[:, :5]    = True
        flg[:, self.nf-5:] = True
        self.wgt[flg] = 0.0

        print("------------------------------\n")

        del temp_vis, temp_flg, temp_wgt, flg
        if self.vlbi == "evn":
            del antenna1, antenna2
        if self.model:
            del model_data
        gc.collect()

    def plot_phase_local(self, ant1, ant2, tini, tend, vis_cal=None, vis_wgt=None, showw=False):

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
        plot_phase(phase_global, wgt_global, time_loc, self.freq,\
                   nbins_time=len(time_loc), nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle, showw=showw)

    def plot_phase_global(self, ant1, ant2, npol, Model=False, tunits = 3600, savef=False, fld=0):
        baseline_d = np.linalg.norm(self.posant[ant2] - self.posant[ant1])
        if len(npol)==1:
            print(f"1 Pol: {npol}")
            vis_global, wgt_global, cal_global, mod_phase = self.load_baseline(ant1, ant2, npol=npol, Data=True, Wgt=True, Cal=True, calmod=True, fld=fld)
            if Model:
                vis_model = self.load_baseline(ant1, ant2, npol=npol, Data=False, Wgt=False, Cal=False, calmod=False, Model=Model, fld=fld)
        else:
            vis_global1, wgt_global1, cal_global1, mod_phase = self.load_baseline(ant1, ant2, npol=npol[0], Data=True, Wgt=True, Cal=True, calmod=True, fld=fld)
            vis_global2, wgt_global2, cal_global2 = self.load_baseline(ant1, ant2, npol=npol[1], Data=True, Wgt=True, Cal=True, calmod=False, fld=fld)
            vis_global = 0.5*(vis_global1+vis_global2)
            cal_global = 0.5*(cal_global1+cal_global2)
            wgt_global = 0.5*(wgt_global1+wgt_global2)
            if Model:
                vis_model1 = self.load_baseline(ant1, ant2, npol=npol[0], Data=False, Wgt=False, Cal=False, calmod=False, Model=Model, fld=fld)[0]
                vis_model2 = self.load_baseline(ant1, ant2, npol=npol[1], Data=False, Wgt=False, Cal=False, calmod=False, Model=Model, fld=fld)[0]
                vis_model = 0.5*(vis_model1+vis_model2)
        
        vis_phase = np.angle(vis_global)
        cal_phase = np.angle(cal_global)
        if Model:
            vis_phase = vis_phase - np.angle(vis_model)
            cal_phase = cal_phase - np.angle(vis_model)

        temp = self.data.query(f'DATA_DESC_ID == {self.SpW} AND FIELD_ID == {fld} AND ANTENNA1 == {ant1} AND ANTENNA2 == {ant2}', columns="TIME")
        time = np.unique(temp.getcol('TIME'))
        time -= time[0]
        temp.close()

        baselinetitle = f"{self.nameant[ant1]} - {self.nameant[ant2]}"
        plot_phase(vis_phase, wgt_global, time/tunits, self.freq,\
                   nbins_time=self.nt, nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle + f" ({baseline_d/1000:.2f} K{self.disunit}) [Original]", showw=False, savef=savef)
        plot_phase(cal_phase, wgt_global, time/tunits, self.freq,\
                   nbins_time=self.nt, nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle + " [Calibrated]", showw=False, savef=savef)
        plot_phase(mod_phase, None, time/tunits, self.freq,\
                   nbins_time=self.nt, nbins_freq=self.nf, nbins_phase = 50, Baselinetitle=baselinetitle + " [Model]", showw=True, savef=savef)

class FringeFitting:
    def __init__(self, ms, nscale=8, tunit=1e3, funit=1e9):
        """
        ms: LOAD_MS class object
        nscale: padding factor for FFT
        tunit: time unit conversion (e.g., to ms)
        funit: freq unit conversion (e.g., to MHz or GHz)
        """

        self.nscale = nscale
        self.tunit  = tunit
        self.funit  = funit
        self.params = np.zeros((ms.nant, 3))
        self.sumw_  = np.zeros((ms.nant))
        self.snr_   = np.zeros((ms.nant))
        self.snr1_  = np.zeros((ms.nant))

        # Constrains
        self.min_delay = self.funit * 0.5/(ms.freq[-1]-ms.freq[0])
        self.max_delay = self.funit * 0.5/ms.df
        self.min_rate  = self.tunit * 0.5/(ms.time[ms.time_edgs[1]] - ms.time[ms.time_edgs[0]])
        self.max_rate  = self.tunit * 0.5/ms.dt

    def FFT_init_ij(self, ms, ant, nt_loc, fringe_rate, delay):
        """Compute fringe-rate, delay, and phase estimates over time-frequency blocks."""

        ant1, ant2 = (ant, ms.refant) if ms.refant > ant else (ms.refant, ant)
        sgn = -1 if ant < ms.refant else 1

        bl_ij = baseline_ij(ant1, ant2, ms.selfbls)

        if ms.baseline_ok[bl_ij]:

            tt1 = bl_ij*nt_loc
            tt2 = tt1 + nt_loc

            # 2D FFT
            F_sub = fftshift(fft2(ms.wgt[tt1:tt2]*ms.vis[tt1:tt2], s=[self.nscale*nt_loc, self.nscale*ms.nf]))
            amp_loc_sh = F_sub.shape

            #IMPLEMENT CONSTRAINS
            F_sub[amp_loc_sh[0]//2, :] = 0.0
            F_sub[:, amp_loc_sh[1]//2] = 0.0

            ampli_loc = np.abs(F_sub)

            # Find peak
            r_idx, tau_idx = np.unravel_index(np.argmax(ampli_loc), amp_loc_sh)

            # if ant1==2 and ant2==25:
            #     plt.figure()
            #     plt.imshow(np.angle(ms.vis[tt1:tt2]), cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi)
            #     plt.figure()
            #     plt.imshow(ampli_loc, cmap="viridis", origin="lower")
            #     plt.axhline(r_idx, ls="--", lw=0.5)
            #     plt.axvline(tau_idx, ls="--", lw=0.5)
            #     plt.show()

            self.params[ant, 0] = sgn*np.angle(F_sub[r_idx, tau_idx])

            rate_ini  = sgn*fringe_rate[r_idx] * self.tunit
            delay_ini = sgn*delay[tau_idx] * self.funit

            # -------------------------------------------------------------------------------------------------#
            # if (ant1==0 and ant2==63):
            #     print(ms.vis[tt1:tt2].shape)

            #     plt.figure(figsize=(8, 6))
            #     plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
            #     plt.imshow(ampli_loc.T, aspect='auto', origin='lower', norm="log",
            #                extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]], vmax=4, vmin=1e-5,
            #                cmap='plasma')
            #     plt.xlabel("Fringe Rate (Hz)")
            #     plt.ylabel(r"Delay ($\mu$s)")
            #     plt.colorbar(label="Amplitude")
            #     plt.show()

            #     plt.figure(figsize=(8, 6))
            #     plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
            #     plt.imshow(np.angle(ms.vis[tt1:tt2]).T, aspect='auto', origin='lower', cmap="twilight",
            #             )
            #     plt.xlabel("Time")
            #     plt.ylabel(r"Freq")
            #     plt.colorbar(label="Phase")
            #     plt.show()

            # if np.abs(rate_ini) < self.min_rate:
            #     rate_ini = sgn_r*self.min_rate
            # elif np.abs(rate_ini) > self.max_rate:
            #     rate_ini = sgn_r*self.max_rate

            # if np.abs(delay_ini) < self.min_delay:
            #     delay_ini = sgn_d*self.min_delay
            # elif np.abs(delay_ini) > self.max_delay:
            #     delay_ini = sgn_d*self.max_delay
            # ----------------------------------------------------------------------------------------------- #

            self.sumw_[ant] = np.sum(ms.wgt[tt1:tt2])
            sumww_ = np.sum(ms.wgt[tt1:tt2]**2)
            xcount_ = np.sum(ms.wgt[tt1:tt2]>0)

            self.snr_[ant] = snr_aips(np.abs(F_sub[r_idx, tau_idx]), self.sumw_[ant], sumww_, xcount_)
            self.snr1_[ant] = SNR(ampli_loc)

            # if self.snr1_[ant] <= 5.0:
            #     rate_ini  = sgn*self.min_rate
            #     delay_ini = sgn*self.min_delay
                # for mant in range(ms.nant):
                #     if mant == ms.refant or mant == ant:
                #         continue
                #     b_ij = baseline_ij(ant, mant, ms.selfbls)
                #     ms.baseline_ok[b_ij] = False
        else:
            rate_ini  = sgn*self.min_rate
            delay_ini = sgn*self.min_delay

        self.params[ant, 1] = rate_ini   
        self.params[ant, 2] = delay_ini

        print(f"Baseline: {ant1}-{ant2} \t phase: {self.params[ant, 0]:.2f} rad \t rate: {self.params[ant, 1]:.2f} mHz \t delay: {self.params[ant, 2]:.2f} ns \t snr_mean {self.snr1_[ant]:.2f}\t snr_aips: {self.snr_[ant]:.2f}")

    def Baseline_FFT_ij(self, ms, iant, jant, nt_loc, fringe_rate, delay):
        """Compute fringe-rate, delay, and phase estimates over time-frequency blocks."""

        ant1, ant2 = (iant, jant) if jant > iant else (jant, iant)
        sgn = -1 if jant < iant else 1

        bl_ij = baseline_ij(ant1, ant2, ms.selfbls)

        # if ant1==2 and ant2==25:
        #     print(ms.baseline_ok[bl_ij])

        if ms.baseline_ok[bl_ij]:

            tt1 = bl_ij*nt_loc
            tt2 = tt1 + nt_loc

            # 2D FFT
            F_sub = fftshift(fft2(ms.vis[tt1:tt2], norm='ortho', s=[self.nscale*nt_loc, self.nscale*ms.nf]))
            amp_loc_sh = F_sub.shape

            #IMPLEMENT CONSTRAINS
            F_sub[amp_loc_sh[0]//2, :] = 0.0
            F_sub[:, amp_loc_sh[1]//2] = 0.0

            ampli_loc = np.abs(F_sub)

            # Find peak
            r_idx, tau_idx = np.unravel_index(np.argmax(ampli_loc), amp_loc_sh)

            self.blparams[3*bl_ij + 0] = sgn*np.angle(F_sub[r_idx, tau_idx])

            rate_ini  = sgn*fringe_rate[r_idx] * self.tunit
            delay_ini = sgn*delay[tau_idx] * self.funit

            # -------------------------------------------------------------------------------------------------#
            # if (ant1==0 and ant2==63):
            #     print(ms.vis[tt1:tt2].shape)

            #     plt.figure(figsize=(8, 6))
            #     plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
            #     plt.imshow(ampli_loc.T, aspect='auto', origin='lower', norm="log",
            #                extent=[fringe_rate[0], fringe_rate[-1], delay[0], delay[-1]], vmax=4, vmin=1e-5,
            #                cmap='plasma')
            #     plt.xlabel("Fringe Rate (Hz)")
            #     plt.ylabel(r"Delay ($\mu$s)")
            #     plt.colorbar(label="Amplitude")
            #     plt.show()

            #     plt.figure(figsize=(8, 6))
            #     plt.title(f"Delay-Fringe Spectrum for Baseline {ant1}-{ant2}")
            #     plt.imshow(np.angle(ms.vis[tt1:tt2]).T, aspect='auto', origin='lower', cmap="twilight",
            #             )
            #     plt.xlabel("Time")
            #     plt.ylabel(r"Freq")
            #     plt.colorbar(label="Phase")
            #     plt.show()

            # if np.abs(rate_ini) < self.min_rate:
            #     rate_ini = sgn_r*self.min_rate
            # elif np.abs(rate_ini) > self.max_rate:
            #     rate_ini = sgn_r*self.max_rate

            # if np.abs(delay_ini) < self.min_delay:
            #     delay_ini = sgn_d*self.min_delay
            # elif np.abs(delay_ini) > self.max_delay:
            #     delay_ini = sgn_d*self.max_delay
            # ----------------------------------------------------------------------------------------------- #

            self.sumw_[iant] = np.sum(ms.wgt[tt1:tt2])
            sumww_ = np.sum(ms.wgt[tt1:tt2]**2)
            xcount_ = amp_loc_sh[0]*amp_loc_sh[1]

            self.snr_[iant] = snr_aips(np.abs(F_sub[r_idx, tau_idx]), self.sumw_[iant], sumww_, xcount_)
            self.snr1_[iant] = SNR(ampli_loc)

            # if self.snr1_[iant] <= 5.0:
            #     rate_ini  = sgn*self.min_rate
            #     delay_ini = sgn*self.min_delay
                # for mant in range(ms.nant):
                #     if mant == ms.refant or mant == ant:
                #         continue
                #     b_ij = baseline_ij(ant, mant, ms.selfbls)
                #     ms.baseline_ok[b_ij] = False
        else:
            rate_ini  = sgn*self.min_rate
            delay_ini = sgn*self.min_delay

        self.blparams[3*bl_ij + 1] = rate_ini   
        self.blparams[3*bl_ij + 2] = delay_ini

        # print(f"Baseline: {ant1}-{ant2} \t phase: {self.blparams[3*bl_ij + 0]:.2f} rad \t rate: {self.blparams[3*bl_ij + 1]:.2f} mHz \t delay: {self.blparams[3*bl_ij + 1]:.2f} ns \t snr_mean {self.snr1_[iant]:.2f}\t snr_aips: {self.snr_[iant]:.2f}")

    def FFT_init_global(self, ms, tint):
        """Compute fringe-rate, delay, and phase estimates over time-frequency blocks for all baselines w.r.t. refant."""
        print("\n ------- INITIAL GUESS ------- ")
        print("Constrains")
        print(f"RATE: min: {self.min_rate:.2f} mHz \t max: {self.max_rate:.2f} mHz")
        print(f"DELAY: min: {self.min_delay:.2f} ns \t max: {self.max_delay:.2f} ns")
        print("------------------------------\n")
        nt_loc      =  ms.time_edgs[tint+1]-ms.time_edgs[tint]
        fringe_rate = fftshift(fftfreq(self.nscale * nt_loc, ms.dt))  # Hz
        delay       = fftshift(fftfreq(self.nscale * ms.nf, ms.df))
        for iant in range(ms.nant):
            if (ms.refant == iant):
                continue
            self.FFT_init_ij(ms, iant, nt_loc, fringe_rate, delay)
        print("Initial Guess set successfully\n")

    def Global_Baseline_FF(self, ms, tint):
        """Compute fringe-rate, delay, and phase estimates over time-frequency blocks for all baselines."""
        print("\n ------- INITIAL GUESS ------- ")
        print("Constrains")
        print(f"RATE: min: {self.min_rate:.2f} mHz \t max: {self.max_rate:.2f} mHz")
        print(f"DELAY: min: {self.min_delay:.2f} ns \t max: {self.max_delay:.2f} ns")
        print("------------------------------\n")
        nt_loc      =  ms.time_edgs[tint+1]-ms.time_edgs[tint]
        fringe_rate = fftshift(fftfreq(self.nscale * nt_loc, ms.dt))  # Hz
        delay       = fftshift(fftfreq(self.nscale * ms.nf, ms.df))
        self.blparams = np.zeros(3*ms.n_bls)
        for iant in range(ms.nant):
            for jant in range(iant+1, ms.nant):
                self.Baseline_FFT_ij(ms, iant, jant, nt_loc, fringe_rate, delay)
        print("Initial Guess set successfully\n")

    def S3(self, vis_loc, wgt_loc, Dt, ms, prms):
        """
        Compute the sum-of-squares cost function between model and measured visibilities.

        Parameters:
            vis_loc (ndarray): Observed visibility data for a baseline.
            wgt_loc (ndarray): Weights for a baseline.
            Dt (ndarray) : Time array (1D).
            Df (ndarray) : Frequency array (1D).
            prms (array): Parameters for baseline [phase, rate, delay].

        Returns:
            float: Sum of squared weighted- phase errors.
        """
        phi0  = wrap_phase(prms[0])
        rDt   = (prms[1]/self.tunit)*Dt
        tauDf = (prms[2]/self.funit)*ms.Dfrq

        Eijk = np.exp(1j*(phi0 + M_PI2*np.add.outer(rDt,tauDf)))
        S2_t = np.abs(vis_loc-Eijk)**2
        S    = np.sum(wgt_loc*S2_t)
    
        return S

    def GlobalCost(self, params, ms, nt_loc, time_n):
        """
        Cost function for least-squares optimization of fringe parameters across all baselines.

        Parameters:
            params (ndarray): Flattened array of parameters [phi0, r, tau] for all antennas.
            ms: LOAD_MS class object
            nt_local: Number of timesteps for the current interval
            time_n: Time array for the current interval.

        Returns:
            float: Sum of squared residuals across all three baselines.
        """

        x0 = np.zeros(3*ms.n_bls)
        for iant in range(ms.nant):
            if iant < ms.refant:
                prmsi = params[3*iant:3*(iant+1)]
            elif iant > ms.refant:
                prmsi = params[3*(iant-1):3*iant]
            else:
                prmsi = [0., 0., 0.]
            for jant in range(iant+1, ms.nant):
                if jant < ms.refant:
                    prmsj = params[3*jant:3*(jant+1)]
                elif jant > ms.refant:
                    prmsj = params[3*(jant-1):3*jant]
                else:
                    prmsj = [0., 0., 0.]
                bl_ij = baseline_ij(iant, jant, ms.selfbls)
                if ~ms.baseline_ok[bl_ij]:
                    continue
                x0[3*bl_ij:3*(bl_ij+1)] = prmsj - prmsi

        S = 0.0
        for bl_ij in range(ms.n_bls):
            if not ms.baseline_ok[bl_ij]:
                continue
            tt1 = bl_ij*nt_loc
            tt2 = tt1 + nt_loc
            vis_loc = ms.vis[tt1:tt2]
            wgt_loc = ms.wgt[tt1:tt2]
            prm_loc = x0[3*bl_ij:3*bl_ij+3]
            S += self.S3(vis_loc, wgt_loc, time_n, ms, prm_loc)
            # S = max(S1, S)
        return S

    def LSM_FF(self, ms, tint, maxiter=50, lsm_="local"):
        print("\n ------- GLOBAL LSM ------- ")
        
        low_bounds = []
        upp_bounds = []
        for _ in range(ms.nant-1):
            low_bounds.extend([-M_PI,-self.max_rate, -self.max_delay])  # phi0, r, tau
            upp_bounds.extend([ M_PI, self.max_rate,  self.max_delay])  # phi0, r, tau

        x0   = self.params.flatten()
        x0   = np.delete(x0, slice(3*ms.refant, 3*(ms.refant+1)))

        tini = ms.time_edgs[tint]
        tend = ms.time_edgs[tint+1]
        nt_loc   =  tend-tini
        time_new = ms.time[tini:tend]-ms.time[tini]
        Err0 = self.GlobalCost(x0, ms, nt_loc, time_new)
        if lsm_ == "local":
            bounds  = np.array([low_bounds, upp_bounds]).T
            result  = least_squares(self.GlobalCost, x0, bounds=(low_bounds, upp_bounds), args=(ms, nt_loc, time_new), max_nfev=maxiter, verbose=2)
            # result  = minimize(self.GlobalCost, x0, args=(ms, nt_loc, time_new), method='L-BFGS-B', bounds=bounds, options={'maxiter':maxiter})
            end_res = result.fun[0]
        elif lsm_ == "global":
            bounds  = np.array([low_bounds, upp_bounds]).T
            print("Sorry, global is not working yet. Hopelly tomorrow it'll be running")
            result  = differential_evolution(self.GlobalCost, x0=x0, args=(ms, nt_loc, time_new), bounds=bounds, maxiter=maxiter)
            end_res = result.fun
        else:
            print(f"lsm: {lsm_} not defined. Must be either 'local' or 'global'")
            exit()
        xn = result.x
        Errn = self.GlobalCost(xn, ms, nt_loc, time_new)
        print(f"residual: {end_res:.2f}")
        print(f"Relative residual change: {(Err0 - Errn)/Err0:.3f}")
        x0_full = np.insert(xn, 3*ms.refant, [0., 0., 0.])

        self.params = x0_full.reshape(ms.nant, 3)

        print(f"LSM is finished. Max. iteration is {maxiter}")
        print("------------------------------\n")


    def calibrate(self, tini, tend, ms):
        """ Save calibrated data in MS table"""
        tloc     = tend-tini
        t_offset = tini*ms.n_bls
        t_limit  = tloc*ms.n_bls
        
        print("\n ------- DATA CALIBRATION ------- ")

        cols = "DATA,FLAG"

        Tab = ms.data.query(query=f"DATA_DESC_ID == {ms.SpW}", offset=t_offset, limit=t_limit, columns=cols)
        temp_vis = Tab.getcol('DATA')
        temp_flg = Tab.getcol('FLAG')

        n_rows   = Tab.nrows()
        Tab.close()

        Tab_cal  = table(ms.ms_path, readonly=False)

        if 'CALIBRATED_DATA' not in Tab_cal.colnames():
            data_desc = Tab_cal.getcoldesc('DATA')
            data_desc['columnname'] = 'CALIBRATED_DATA'
            Tab_cal.addcols({'CALIBRATED_DATA': data_desc})
        if 'CALIBRATED_MODEL' not in Tab_cal.colnames():
            desc = makearrcoldesc(columnname='CALIBRATED_MODEL',
                          value=0.,
                          valuetype='float',
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
                    bl_ij = baseline_ij(iant, jant, ms.selfbls)
                    if ~ms.baseline_ok[bl_ij]:
                        continue
                    prms = self.params[jant] - prmsi
                    if ms.bl_based:
                        prms = self.blparams[3*bl_ij: 3*bl_ij+3]
                    phase_mod = prms[0]+M_PI2*(prms[1]/self.tunit*(ms.time[tini+tt]-ms.time[tini])
                                             + prms[2]/self.funit*(ms.freq-ms.freq[0]))
                    G_ff_inv = np.exp(-1j*phase_mod)

                    row_idx = ms.order_bls[bl_ij] + tt*ms.n_bls
                    for ii in range(temp_vis.shape[-1]):
                        calibrated_vis[row_idx, :, ii] = temp_vis[row_idx, :, ii]*G_ff_inv
                    calibrated_mod[row_idx] = np.angle(G_ff_inv)

        Tab_out = Tab_cal.query(query=f"DATA_DESC_ID == {ms.SpW}", offset=t_offset, limit=t_limit, columns="CALIBRATED_DATA,CALIBRATED_MODEL")
        Tab_out.putcol('CALIBRATED_DATA', calibrated_vis)
        Tab_out.putcol('CALIBRATED_MODEL', calibrated_mod)
        Tab_out.close()
        Tab_cal.close()

        # new_blij = ms.order_bls[baseline_ij(0, 72, ms.selfbls)]
        # plt.figure()
        # plt.imshow(np.angle(calibrated_vis[new_blij::ms.n_bls, :, 0]).T, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="auto", origin="lower")
        # plt.figure()
        # plt.imshow(calibrated_mod[new_blij::ms.n_bls, :].T, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="auto", origin="lower")
        # plt.show()

        del temp_vis, temp_flg, calibrated_vis, calibrated_mod

        gc.collect()

        print("Calibration complete")
        print("------------------------------\n")

    def save_prmts(self, name='params'):
        np.save(f"{name}.npy", self.params)

    def load_prmts(self, name='params'):
        self.params = np.load(f"{name}.npy", allow_pickle=True)

if __name__ == "__main__":

    print("\n     *** FRINGE FITTING CODE ***\n\n")
    # Path to the MeasurementSet (MS)
    # data_MS = "../Data/ILTJ123441.23+314159.4_141MHz_uv.dp3-concat"
    # data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
    data_MS = "../Data/ILTJ131028.61+322045.7_143MHz_uv.dp3-concat"
    # data_MS = "../Data/n24l2.ms"

    refant      = 24
    npol        = [0, 3]
    nSpW        = 0
    tintervals  = 16
    nscaleg     = 64
    LSM         = False
    lsm_        = "local"
    selfbls     = False
    vlbi        = "lofar"
    uvrange     = 2e4
    modl        = True
    Bl_based    = True

    # refant      = 2
    # npol        = [0, 3]
    # nSpW        = 1
    # tintervals  = 8
    # nscaleg     = 64
    # LSM         = True
    # lsm_        = "local"
    # selfbls     = True
    # vlbi        = "evn"
    # uvrange     = 0
    # modl        = False
    # Bl_based    = False

    ms_data     = LOAD_MS(data_MS, npol=npol, SpW=nSpW, refant=refant, tints=tintervals, selfbls=selfbls, vlbi=vlbi, model=modl, bl_based=Bl_based)

    #### START MAIN LOOP
    for tint in range(ms_data.tints):
        print(f"\n---> Time interval = {tint+1}/{ms_data.tints}")
        tini = ms_data.time_edgs[tint]
        tend = ms_data.time_edgs[tint+1]
        print(f"Time steps: {tend-tini}\n")
        ms_data.flag_baselines(tini, tend, uvrange = uvrange)

        ms_data.load_data(tini, tend, corrcomb=2)

        Global_FF = FringeFitting(ms_data, nscale=nscaleg)
        if not Bl_based:
            Global_FF.FFT_init_global(ms_data, tint)
        else:
            Global_FF.Global_Baseline_FF(ms_data, tint)
        print("TIME: ", tm.time()-INI_TIME)

        # #### PLOT FLAGGED DATA
        # baseline_test = np.zeros([ms_data.nant, ms_data.nant])

        # for ii in range(ms_data.nant):
        #     for jj in range(ii+1, ms_data.nant):
        #         blij = baseline_ij(ii, jj, ms_data.selfbls)
        #         baseline_test[ii, jj] = ms_data.baseline_ok[blij]
        #         baseline_test[jj, ii] = ms_data.baseline_ok[blij]

        # plt.figure(figsize=(8, 6))
        # plt.imshow(baseline_test, origin='lower')
        # plt.xlabel("Antenna 2")
        # plt.ylabel("Antenna 1")
        # plt.colorbar(label='% Flagged Data')
        # plt.grid(False)
        # plt.show()

        # ##########################

        if LSM:
            Global_FF.LSM_FF(ms_data, tint, maxiter=100, lsm_=lsm_)
            print("TIME: ", tm.time()-INI_TIME)

        Global_FF.calibrate(tini, tend, ms_data)

        print("TIME: ", tm.time()-INI_TIME)
 
    plt.figure(figsize=(10, 6))
    # for tint in range(Global_FF.tints):
    plt.plot(Global_FF.snr_, ".")
    plt.xlabel("Antenna 2")
    plt.ylabel("SNR_AIPS")
    plt.grid(False)

    plt.figure(figsize=(10, 6))
    # for tint in range(Global_FF.tints):
    plt.plot(Global_FF.snr1_, ".")
    plt.xlabel("Antenna 2")
    plt.ylabel("SNR_MEAN")
    plt.grid(False)

    plt.figure(figsize=(10, 6))
    # for tint in range(Global_FF.tints):
    plt.plot(Global_FF.sumw_, ".")
    plt.xlabel("Antenna 2")
    plt.ylabel("SUMW")
    plt.grid(False)
    plt.show()

    ms_data.close()
    print('\n                   *** END ***\n')
    print('Come back tomorrow Buddy...')
    print('...perhabs the code will work smoothly by then :V\n')