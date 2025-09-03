// Load DATA using Casacore:: MeasurementSet
// Calculate Initial State guess using FFT
// Applying callibration

#include <casacore/ms/MeasurementSets.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/TaQL/TableParse.h>
#include <casacore/tables/Tables/TableDesc.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableRecord.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Vector.h>

#include <complex>
#include <iostream>
#include <memory>
#include <fftw3.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xfunctor_view.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include <chrono>

typedef xt::xarray<double> time_type;
typedef xt::xarray<float> frequency_type;
typedef xt::xarray<bool> flag_type;
typedef xt::xarray<std::complex<float>> visibility_type;
typedef xt::xarray<float> weight_type;
typedef xt::xarray<float> params_type;

using namespace casacore;

const double TUNIT  = 1e3;
const double FUNIT  = 1e9;

void read_visibilities_order(MeasurementSet &ms, visibility_type &visibilities, weight_type &weights, uint tini, uint tend, uint n_bls) {
  uint row_ini = tini*n_bls;
  uint row_end = tend*n_bls;
  uint tskip = tend-tini;
  std::cout << "\n READING DATA" << std::endl;
  std::ostringstream query;
  query << "select from $1 LIMIT " << row_ini << ":" << row_end << ":1";
  casacore::TaQLResult TabData = casacore::tableCommand(query.str(), ms);

  ArrayColumn<Complex> dataCol(TabData.table(), "DATA");
  ArrayColumn<Float> wgtCol(TabData.table(), "WEIGHT_SPECTRUM");
  ArrayColumn<Bool> flagsCol(TabData.table(), "FLAG");

  auto shape = dataCol.shape(0);
  auto nrows = dataCol.nrow();
  uint nchannels = shape[1];
  uint npol = shape[0];

  if (nrows != tskip * n_bls) {
    std::cerr << "Warning: Expected " << tskip * n_bls << " rows but got " << nrows << std::endl;
  }
    
  // Now resize final arrays for baseline-major order
  visibilities.resize({nrows, nchannels, npol});
  weights.resize({nrows, nchannels, npol});
  xt::xarray<bool> flags;
  flags.resize({nrows, nchannels, npol});
    
  // Read data row by row into temporary arrays
  uint irow, pol, chan, bl, tt, row_new;

  for (irow = 0; irow < nrows; ++irow) {
    Array<Complex> vis_data = dataCol.get(irow);
    Array<Float> weight_data = wgtCol.get(irow);
    Array<Bool> flag_data = flagsCol.get(irow);
    bl = irow % n_bls;
    tt = irow / n_bls;
    row_new = bl * tskip + tt;
    for (pol = 0; pol < npol; ++pol) {
      for (chan = 0; chan < nchannels; ++chan) {
        IPosition pos(2, pol, chan);
        float phase = std::arg(vis_data(pos));  
        visibilities(row_new, chan, pol) = std::polar(1.0f, phase);  // only phase
        // visibilities(row_new, chan, pol) = std::complex<float>(vis_data(pos));
        std::complex<float> phasor = std::polar(1.0f, phase);
        
        bool flg = flag_data(pos);
        // enforce edge channel flags
        if (chan < 5 || chan >= nchannels - 5) {
            flg = true;
        }

        // apply masking
        if (flg) {
            visibilities(row_new, chan, pol) = {0.0f, 0.0f};
            weights(row_new, chan, pol) = 0.0f;
        } else {
            visibilities(row_new, chan, pol) = phasor;
            weights(row_new, chan, pol) = weight_data(pos);
        }

        // store final flags
        flags(row_new, chan, pol) = flg;
        }
    }

    if (irow % 10000 == 0) {
      std::cout << "Read " << irow << " rows..." << std::endl;
    }
  }
    
  std::cout << "Applying flags..." << std::endl;
  xt::filtration(visibilities, flags) = std::complex<float>{0, 0};
  xt::filtration(weights, flags) = 0.0;
  std::cout << "Data reordering complete.\n" << std::endl;
}



uint num_time_steps(const MeasurementSet &ms) {
  uint nt = tableCommand("select from $1 orderby unique TIME", ms).table().nrow();
  return nt;
}

uint num_antennas(const MeasurementSet &ms) {
  uint nant = ms.keywordSet().asTable("ANTENNA").nrow();
  return nant;
}

uint num_baselines(const MeasurementSet &ms) {
  uint nrbls = tableCommand("select unique ANTENNA1,ANTENNA2 from $1", ms).table().nrow();
  return nrbls;
}

uint baselineij(uint iant, uint jant) {
  uint bl_ij = (jant*(jant-1))/2 + iant;
  return bl_ij;
}

void read_frequencies(MeasurementSet &ms, frequency_type &frequencies) {
  Table tab(ms.spectralWindowTableName());
  ArrayColumn<Double> frequencyCol(tab, "CHAN_FREQ");
  Array<Double> channel_frequencies = frequencyCol.get(0);
  auto shape = frequencyCol.shape(0);
  uint nfreqs = shape[0];
  frequencies.resize({nfreqs});
  xt::xarray<double> tmp_frequencies;
  tmp_frequencies.resize({nfreqs});
  memcpy(tmp_frequencies.data(), channel_frequencies.data(),
         sizeof(double) * nfreqs);
  frequencies = xt::cast<float>(tmp_frequencies);
}

void read_time(MeasurementSet &ms, time_type &time, uint n_corr) {
  Table tab(ms);
  ScalarColumn<Double> timeCol(tab, "TIME");
  uint nrows = timeCol.nrow();
  uint nt = nrows/n_corr;
  xt::xarray<double> tmp_time;
  tmp_time.resize({nt});
  int ii;
  double t0 = timeCol(0);
  for (ii = 0; ii < nt; ++ii) {
    tmp_time(ii) = static_cast<double>(timeCol(ii*n_corr)) - t0;
  }
  time = tmp_time;
}

void filter_baseline(MeasurementSet &ms, flag_type &baseline_ok, uint tini, uint tend, uint n_bls, double threshold=0.9){
  uint row_ini = tini*n_bls;
  uint row_end = tend*n_bls;
  uint tskip = tend-tini;
  std::ostringstream query;
  query << "select from $1 LIMIT "<<row_ini<<":"<<row_end<<":"<<"1";
  casacore::TaQLResult TabData = casacore::tableCommand(query.str(), ms);
  ArrayColumn<Bool> flagCol(TabData.table(), "FLAG");
  uint nrows = flagCol.nrow();
  auto shape = flagCol.shape(0);
  uint totdata = shape[0]*shape[1]*tskip;
  baseline_ok.resize({n_bls});
  int ti, bl;
  std::vector<bool> data;
  std::vector<float> sum(n_bls, 0.0);
  int Tot = 0;
  for (ti=0; ti < tskip; ++ti) {
    for (bl=0; bl < n_bls; ++bl){
      data = flagCol.get(n_bls*ti + bl).tovector();
      int row_sum = std::accumulate(data.begin(), data.end(), 0);
      sum[bl] += row_sum;
    }
  }
  for (bl=0; bl < n_bls; ++bl){
    baseline_ok[bl] = sum[bl]/totdata < threshold;
    if (baseline_ok[bl]){Tot ++;}
  }
  std::cout << "Active Baselines:\t" << Tot << std::endl;
}

void first_guess(
  params_type &params, visibility_type &visibilities, weight_type &weights, flag_type &baseline_ok, time_type &time,
  frequency_type &freq, double f0_, uint nant, uint refant, uint nt, uint nf, uint nscale)
{
  fftw_complex *F_tf, *FT_rd;
  fftw_plan plan;

  uint nt_new = nt*nscale;
  uint nf_new = nf*nscale; 

  F_tf  = (fftw_complex*) fftw_malloc(nt_new * nf_new * sizeof(fftw_complex));
  FT_rd = (fftw_complex*) fftw_malloc(nt_new * nf_new * sizeof(fftw_complex));

  plan = fftw_plan_dft_2d(nt_new, nf_new, F_tf, FT_rd, FFTW_FORWARD, FFTW_ESTIMATE);

  params.resize({nant, 3});

  double dt = time(1) - time(0);
  double df = freq(1) - freq(0);

  int iant, bl_ij, ti, fi;
  int sgn = 0;

  for (iant = 0; iant < nant; ++iant) {
    if (iant == refant) {
      params(iant, 0) = 0.0;
      params(iant, 1) = 0.0;
      params(iant, 2) = 0.0;
      continue;
    }

    sgn   = (iant > refant) ? 1 : -1;
    bl_ij = (iant > refant) ? baselineij(refant, iant) : baselineij(iant, refant);

    if (!baseline_ok[bl_ij]) continue;

    // Stokes I
    auto vis_slice0 = xt::eval(xt::view(
        visibilities,
        xt::range(bl_ij * nt, (bl_ij + 1) * nt),
        xt::all(),
        0
    ));
    
    auto vis_slice1 = xt::eval(xt::view(
        visibilities,
        xt::range(bl_ij * nt, (bl_ij + 1) * nt),
        xt::all(),
        3
    ));
    
    auto wgt_slice0 = xt::eval(xt::view(
        weights,
        xt::range(bl_ij * nt, (bl_ij + 1) * nt),
        xt::all(),
        0
    ));
    
    auto wgt_slice1 = xt::eval(xt::view(
        weights,
        xt::range(bl_ij * nt, (bl_ij + 1) * nt),
        xt::all(),
        3
    ));
    
    auto vis_slice = xt::eval(0.5f * (vis_slice0 + vis_slice1));
    auto wgt_slice = xt::eval(0.5f * (wgt_slice0 + wgt_slice1));
    
    // Fill F_tf: Zero-padding
    for (ti = 0; ti < nt_new; ++ti) {
      for (fi = 0; fi < nf_new; ++fi) {
        if (ti < nt && fi < nf) {
          auto val = vis_slice(ti, fi);
          std::complex<float> weighted_val = wgt_slice(ti, fi) * val;
          F_tf[ti * nf_new + fi][0] = weighted_val.real();
          F_tf[ti * nf_new + fi][1] = weighted_val.imag();
        } else {
          F_tf[ti * nf_new + fi][0] = 0.0;
          F_tf[ti * nf_new + fi][1] = 0.0;
        }
      }
    }

    fftw_execute(plan);

  // -------------------- DELAY LIMITS --------------------
    double maxDelay =  0.5 / df;   // Max positive delay (seconds)
    double minDelay = -0.5 / df;   // Max negative delay (seconds)

    double bw = df * nf_new;    // Total FFT bandwidth (Hz)

  // Convert delay limits to FFT bin indices
    int i0 = Int(minDelay * bw);
    int i1 = Int(maxDelay * bw);

    if (i0 > i1) std::swap(i0, i1);
    if (i1 == i0) ++i1;

  // -------------------- RATE LIMITS --------------------
    double maxRate =  0.5 / (dt * f0_);
    double minRate = -0.5 / (dt * f0_);

    double width = dt * nt_new * f0_;  // FFT time domain width in Hz

    // Convert rate limits to FFT bin indices
    int j0 = Int(minRate * width);
    int j1 = Int(maxRate * width);

    if (j0 > j1) std::swap(j0, j1);
    if (j1 == j0) ++j1;

  // -------------------- WRAPPED SEARCH LOOPS --------------------
    double max_mag = -1.0;
    uint ipkt = 0;
    uint ipkch = 0;
    uint ipk1d = 0;

    for (int itime0 = j0; itime0 < j1; ++itime0) {
      int itime = (itime0 < 0) ? itime0 + nt_new : itime0;
      for (int ich0 = i0; ich0 < i1; ++ich0) {
        int ich = (ich0 < 0) ? ich0 + nf_new : ich0;
        if (itime == 0 || ich == 0) continue;
        uint idx = itime * nf_new + ich;
        double re = FT_rd[idx][0];
        double im = FT_rd[idx][1];
        double mag = re * re + im * im;
        if (mag > max_mag) {
          max_mag = mag;
          ipkt = itime;
          ipkch = ich;
          ipk1d = idx;
        }
      }
    }

    double peak_re = FT_rd[ipk1d][0];
    double peak_im = FT_rd[ipk1d][1];
    double phase = std::atan2(peak_im, peak_re);

  // --- Delay (Frequency axis) ---
    double delay_norm = double(ipkch)/double(nf_new);

    if (delay_norm > 0.5)  // FFT wrapping correction
      delay_norm -= 1.0;

    double delay = delay_norm / df;  // seconds of delay

  // --- Rate (Time axis) ---
    double rate_norm = double(ipkt) / double(nt_new);

    if (rate_norm > 0.5)  // FFT wrapping correction
      rate_norm -= 1.0;

    double rate = rate_norm / dt;             // Hz

    params(iant, 0) = float(sgn * phase);
    params(iant, 1) = float(sgn * rate)*TUNIT;
    params(iant, 2) = float(sgn * delay)*FUNIT;

    std::cout << " Ant:   " << iant
              << "\tPhase: " << params(iant, 0)
              << "\tRate:  " << params(iant, 1)
              << "\tDelay: " << params(iant, 2)
              << std::endl;
    }

    fftw_destroy_plan(plan);
    fftw_free(F_tf);
    fftw_free(FT_rd);
}

void calibrate_visibilities(
    visibility_type &calibrated_vis,
    const visibility_type &vis,
    const params_type &params,
    const flag_type &baseline_ok,
    const time_type &time,
    const frequency_type &freq,
    uint nf,
    uint tini,
    uint tloc,
    uint nant,
    uint n_bls
) {
    uint npol = vis.shape()[2];
    calibrated_vis.resize({tloc*n_bls, nf, npol});

    for (uint tt = 0; tt < tloc; ++tt) {
      for (uint iant = 0; iant < nant; ++iant) {
        auto prms_i = xt::view(params, iant, xt::all());
        for (uint jant = iant + 1; jant < nant; ++jant) {
          uint bl_ij = baselineij(iant, jant);
          if (!baseline_ok[bl_ij]) continue;
          xt::xarray<float> prms = xt::view(params, jant, xt::all()) - prms_i;
          xt::xarray<double> phase_mod = prms(0) + M_PI * 2.0 * (prms(1)/TUNIT * (time(tt) - time(tini)) + prms(2)/FUNIT * (freq - freq(0)));
          auto phase_mod_f = xt::cast<float>(phase_mod);
          xt::xarray<std::complex<float>> G_ff_inv = xt::exp(std::complex<float>(0.0f, -1.0f) * phase_mod_f);

          uint row_old = bl_ij*tloc + tt;
          uint row_new = tt*n_bls + bl_ij;

          for (uint pol = 0; pol < npol; ++pol) {
            auto vis_row = xt::view(vis, row_old, xt::all(), pol);
            auto cal_row = xt::view(calibrated_vis, row_new, xt::all(), pol);
            // cal_row = xt::eval(vis_row * G_ff_inv);
            cal_row = xt::eval(G_ff_inv);
          }
        }
      }
    }
    std::cout << "Calibration completed!" << std::endl;
}

void write_calibrated_to_ms(
    casacore::MeasurementSet &ms,
    const visibility_type &calibrated_vis,
    uint row_offset  // <-- new
) {
    using namespace casacore;

    // Check if CALIBRATED_DATA column exists, otherwise add it
    TableDesc td = ms.tableDesc();
    if (!td.isColumn("CALIBRATED_DATA")) {

      std::cout << "ERRORRRR" << std::endl;
        // TableDesc newtd = td;
        // newtd.addColumn(ArrayColumnDesc<Complex>(
        //     "CALIBRATED_DATA",
        //     "Calibrated visibilities",
        //     2   // shape [npol, nchan]
        // ));
        // ms.addColumn(newtd.columnDesc("CALIBRATED_DATA"));
    }

    ArrayColumn<Complex> calCol(ms, "CALIBRATED_DATA");

    uint nrows = calibrated_vis.shape()[0];
    uint nchan = calibrated_vis.shape()[1];
    uint npol  = calibrated_vis.shape()[2];

    for (uInt r = 0; r < nrows; ++r) {
        casacore::Array<Complex> rowData(IPosition(2, npol, nchan));
        for (uInt pol = 0; pol < npol; ++pol) {
            for (uInt chan = 0; chan < nchan; ++chan) {
                rowData(IPosition(2, pol, chan)) = calibrated_vis(r, chan, pol);
            }
        }
        calCol.put(row_offset + r, rowData);  // <-- apply offset
    }

    ms.flush();
    std::cout << "Finished writing CALIBRATED_DATA column to MS" << std::endl;
}

int main(void){
  visibility_type vis;
  visibility_type cal_vis;
  weight_type wgt;
  flag_type baseline_ok;
  frequency_type freq;
  time_type time;

  const std::string msin = "../Data/ILTJ123441.23+314159.4_141MHz_uv.dp3-concat";

  std::cout << "Fringe Fitting: Start Point ..." << std::endl;
  MeasurementSet ms(msin);
  
  auto start = std::chrono::high_resolution_clock::now();

  read_frequencies(ms, freq);
  double f0_ = xt::mean(freq)();
  uint nant = num_antennas(ms);
  uint refant = 25;
  uint nf = freq.size();
  uint nscale = 32;
  const uint n_correlations = num_baselines(ms);

  uint nt = num_time_steps(ms);
  uint tintervals = 8; 
  uint dt   = nt/tintervals; 
  
  std::cout << "DATA INFO ..." << std::endl;
  std::cout << "Num_antenna:         " << nant << std::endl;
  std::cout << "Num_channels:        " << nf   << std::endl;
  std::cout << "Time steps:          " << nt   << std::endl;
  std::cout << "Num_correlations:    " << n_correlations << std::endl;
  std::cout << "Time of Observation: " << (time[nt-1] - time[0])/3600 << " Hours" << std::endl;
  std::cout << "Intervals            " << tintervals << std::endl;
  std::cout << "DT                   " << dt   << std::endl;
  
  uint tini = 0;
  uint tend = 0;
  uint nt_loc;
  read_time(ms, time, n_correlations);
  
  int tint;

  MeasurementSet ms_new(msin, casacore::TableLock::AutoNoReadLocking, casacore::Table::Update);

  for (tint=0; tint < tintervals; tint++) {

    tini  = tend;
    tend  = tint == tintervals-1 ? nt : tini + dt;
    nt_loc= tend - tini;

    filter_baseline(ms, baseline_ok, tini, tend, n_correlations);
    read_visibilities_order(ms, vis, wgt, tini, tend, n_correlations);

    std::cout << "Interval " << tint << "\t ti: " << tini << "\t te: " << tend << std::endl;
    std::cout << "FFT: First Guess..." << std::endl;

    auto mid = std::chrono::high_resolution_clock::now();
    params_type prms;

    first_guess(prms, vis, wgt, baseline_ok, time, freq, f0_, nant, refant, nt_loc, nf, nscale);

    // calibration
    calibrate_visibilities(cal_vis, vis, prms, baseline_ok, time, freq, nf, tini, nt_loc, nant, n_correlations);

    uint row_offset = tini * n_correlations;
    write_calibrated_to_ms(ms_new, cal_vis, row_offset);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::chrono::duration<double> elapsed1 = end - mid;

    std::cout << "Total time : " << elapsed.count() << " seconds" << std::endl;
    std::cout << "\nFFT time : " << elapsed1.count() << " seconds" << std::endl;
  }
  std::cout << "Just printing... Nothing new yet." << std::endl;
}