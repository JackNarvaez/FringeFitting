#include <casacore/tables/Tables.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/measures/TableMeasures/TableMeasDesc.h>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>

using namespace casacore;
using namespace std;

typedef complex<double> cdouble;

struct MSInfo {
    vector<double> time;
    vector<double> freq;
    size_t nt, nf;
    double dt, df;
    size_t nant;
    vector<string> nameant;
};

MSInfo loadMSInfo(const string& ms_path) {
    MSInfo info;

    // Time
    Table ms(ms_path);
    Array<double> time_col = ms.getcol("TIME");
    Vector<double> time_vec = time_col.reform(IPosition(1, time_col.shape()[0]));
    set<double> unique_times(time_vec.begin(), time_vec.end());
    info.time = vector<double>(unique_times.begin(), unique_times.end());
    info.nt = info.time.size();
    info.dt = info.time[1] - info.time[0];
    ms.close();

    // Frequency
    Table spw(ms_path + "/SPECTRAL_WINDOW");
    Array<double> chan_freq = spw.getcol("CHAN_FREQ");
    Vector<double> freq_vec = chan_freq.reform(IPosition(1, chan_freq.shape()[1]));
    info.freq = vector<double>(freq_vec.begin(), freq_vec.end());
    info.nf = info.freq.size();
    info.df = info.freq[1] - info.freq[0];
    spw.close();

    // Antennas
    Table ant(ms_path + "/ANTENNA");
    ROScalarColumn<String> name_col(ant, "NAME");
    info.nant = ant.nrow();
    for (size_t i = 0; i < info.nant; ++i) {
        info.nameant.push_back(name_col(i));
    }
    ant.close();

    return info;
}

void printInfo(const MSInfo& info) {
    cout << "Time steps: " << info.nt << ", Δt: " << info.dt << " s" << endl;
    cout << "Freq channels: " << info.nf << ", Δf: " << info.df << " Hz" << endl;
    cout << "Antennas: " << info.nant << endl;
    for (size_t i = 0; i < info.nameant.size(); ++i) {
        cout << "  [" << i << "] " << info.nameant[i] << endl;
    }
}

void runFFT2(const vector<vector<cdouble>>& vis, size_t nscale, vector<vector<double>>& power2d) {
    size_t nt = vis.size();
    size_t nf = vis[0].size();
    size_t nt_fft = nscale * nt;
    size_t nf_fft = nscale * nf;

    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nt_fft * nf_fft);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nt_fft * nf_fft);

    // Fill input
    for (size_t t = 0; t < nt_fft; ++t) {
        for (size_t f = 0; f < nf_fft; ++f) {
            size_t idx = t * nf_fft + f;
            if (t < nt && f < nf) {
                in[idx][0] = vis[t][f].real();
                in[idx][1] = vis[t][f].imag();
            } else {
                in[idx][0] = 0.0;
                in[idx][1] = 0.0;
            }
        }
    }

    // Execute
    fftw_plan plan = fftw_plan_dft_2d(nt_fft, nf_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Compute power spectrum
    power2d.resize(nt_fft, vector<double>(nf_fft, 0.0));
    for (size_t t = 0; t < nt_fft; ++t) {
        for (size_t f = 0; f < nf_fft; ++f) {
            size_t idx = t * nf_fft + f;
            double re = out[idx][0], im = out[idx][1];
            power2d[t][f] = sqrt(re*re + im*im);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

// Placeholder: implement fringe rate & delay estimation
void estimateFringes(const MSInfo& info, const vector<vector<cdouble>>& vis, size_t nscale) {
    vector<vector<double>> power;
    runFFT2(vis, nscale, power);

    // Find peak in power spectrum
    double maxval = 0;
    size_t max_t = 0, max_f = 0;
    for (size_t t = 0; t < power.size(); ++t) {
        for (size_t f = 0; f < power[0].size(); ++f) {
            if (power[t][f] > maxval) {
                maxval = power[t][f];
                max_t = t;
                max_f = f;
            }
        }
    }

    cout << "FFT peak at (t, f): (" << max_t << ", " << max_f << "), Power = " << maxval << endl;

    // Compute delay and rate from indices (TODO: use fftfreq)
    double delay = (max_f - power[0].size() / 2) * 1.0 / (info.df * nscale);
    double rate  = (max_t - power.size() / 2) * 1.0 / (info.dt * nscale);

    cout << "Estimated delay: " << delay << " s" << endl;
    cout << "Estimated rate:  " << rate << " Hz" << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./fringe_fit <MeasurementSet path>" << endl;
        return 1;
    }

    string ms_path = argv[1];

    MSInfo info = loadMSInfo(ms_path);
    printInfo(info);

    // Mock visibility for one baseline (should be read from DATA column)
    vector<vector<cdouble>> vis(info.nt, vector<cdouble>(info.nf, cdouble(1.0, 0.0)));
    
    // Call FFT
    estimateFringes(info, vis, 8);

    return 0;
}
