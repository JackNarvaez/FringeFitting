import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from   casacore.tables import table

data_FD  = "../Data/"
# data_MS  = data_FD + "n24l2.ms"
data_MS  = data_FD + "ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
antennas = table(data_MS+"/ANTENNA")
antname  = antennas.getcol("NAME")
nant     = len(antname)
print("Antenna table contains ", antennas.colnames())
print("# ant: ", nant)
print(antname)

antenna_names = antennas.getcol("NAME")
spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
print("Antenna table contains ", spectral_window.colnames())

ref_frequency = spectral_window.getcol('REF_FREQUENCY')
print("The observation was performed at frequency ", spectral_window[0]["REF_FREQUENCY"], "Hz or ",  spectral_window[0]["REF_FREQUENCY"] / 1.e6 , " MHz")
print("The number of channels available are ", len(spectral_window[0]['CHAN_FREQ']), " at frequency ", spectral_window[0]['CHAN_FREQ']/1.e6 , " MHz")

pol_table = table(data_MS + "/POLARIZATION")
corr_types = pol_table[0]["CORR_TYPE"]
print("Polarizations available:", corr_types)

Pointing = table(data_MS+"/POINTING")
Target = Pointing.getcol("TARGET")
print("Pointing ", Pointing.colnames())

n_spw = len(spectral_window)

for spw_id in range(n_spw):
    ref_freq = spectral_window[spw_id]['REF_FREQUENCY']
    chan_freqs = spectral_window[spw_id]['CHAN_FREQ']
    chan_widths = spectral_window[spw_id]['CHAN_WIDTH']
    total_bw = spectral_window[spw_id]['TOTAL_BANDWIDTH']
    
    print(f"\nSpectral Window {spw_id}:")
    print(f"  Reference Frequency: {ref_freq / 1e6:.3f} MHz")
    print(f"  Total Bandwidth    : {total_bw / 1e6:.3f} MHz")
    print(f"  Number of Channels : {len(chan_freqs)}")
    print(f"  First Channel Freq : {chan_freqs[0] / 1e6:.3f} MHz")
    print(f"  Last Channel Freq  : {chan_freqs[-1] / 1e6:.3f} MHz")
    print(f"  Channel Width      : {chan_widths[0] / 1e3:.3f} kHz")

antennas.close()
spectral_window.close()
pol_table.close()
Pointing.close()

data = table(data_MS)

spW = np.unique(data.getcol("DATA_DESC_ID"))
time = np.unique(data.getcol("TIME"))
plt.plot(np.diff(time), ".")
plt.show()

ntt = len(time)
print("DATA: ", data.colnames())
print("TIME: ", len(data.getcol("TIME")))
print("WGT : ", data.getcol("WEIGHT")[:].shape)
print("Tt  : ", ntt)
print("FD1 : ", np.unique(data.getcol("FEED2")))
print("FD2 : ", np.unique(data.getcol("FEED1")))
print("spW : ", spW)
print("UVW : ", len(data.getcol("UVW")))


##### Check ORDER
n_bls = (nant*(nant-1))//2
check_time = data.query(query=f"DATA_DESC_ID == {1}", columns="ANTENNA1,ANTENNA2")
antenna1 = check_time.getcol("ANTENNA1")
antenna2 = check_time.getcol("ANTENNA2")
check_time.close()

for ti in range(ntt):
    cond1 = np.sum(antenna1[ti*n_bls:(ti+1)*n_bls]!=antenna1[:n_bls])
    cond2 = np.sum(antenna2[ti*n_bls:(ti+1)*n_bls]!=antenna2[:n_bls])
    if (cond1 + cond2) > 0:
        print(ti)

check_t  = data.query(query=f"ANTENNA1 == {10} AND ANTENNA2 == {15}", columns="TIME")
time_bl = check_t.getcol("TIME")
check_time.close()

plt.plot(np.diff(time), ".")
plt.show()

l = (sp.constants.c) / ref_frequency
uvw = data.getcol("UVW") / l

plt.gca().set_aspect('equal', 'box')
plt.scatter(uvw[:,0], uvw[:, 1], s=0.1, c="k")
plt.xlabel("u [$\lambda$]")
plt.ylabel("v [$\lambda$]")
plt.xlim(-8e5, 8e5)
plt.ylim(-8e5, 8e5)
plt.show()

d_uvw = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 2] **2)
clean_data = []

nchan = 10
n_spW = 0

clean_data = [row['DATA'][nchan, 0] for row in data]
amplitude_crosscorrelations = np.abs(clean_data[n_spW::num_spw])
data.close()

plt.scatter(d_uvw, amplitude_crosscorrelations, s=1, c="k")
plt.semilogx()
plt.ylabel("Intensity")
plt.xlabel("Baseline size [$\lambda$]")
plt.show()
