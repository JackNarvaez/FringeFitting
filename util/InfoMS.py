"""
Show information of a Measurement Set
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from   casacore.tables import table

data_FD  = "./Data/"
# data_MS  = data_FD + "n24l2.ms"
data_MS  = data_FD + "ILTJ123441.23+314159.4_141MHz_uv.dp3-concat"
# data_MS  = data_FD + "ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
# data_MS  = data_FD + "lofar-ds1-storage.ms"
# data_MS  = data_FD + "ILTJ131028.61+322045.7_143MHz_uv.dp3-concat"

antennas = table(data_MS+"/ANTENNA")
antname  = antennas.getcol("NAME")
antpos   = antennas.getcol("POSITION")
nant     = len(antname)
print("Antenna table contains ", antennas.colnames())
print("# ant: ", nant)
print(antennas.getcol("STATION"))
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

uvw_bls = np.zeros([nant, nant])
for ii in range(nant):
    for jj in range(ii+1, nant):
        uvw_bls[ii, jj] = np.linalg.norm(antpos[jj] - antpos[ii])
        uvw_bls[jj, ii] = uvw_bls[ii, jj]

plt.imshow(uvw_bls, origin="lower")
plt.colorbar()
plt.show()
plt.figure()
plt.plot(uvw_bls, ".")
plt.yscale("log")
plt.show()

uv_max_i, uv_max_j = np.unravel_index(np.argmax(uvw_bls), uvw_bls.shape)
print(f"\nMAXUV = {np.max(uvw_bls)}, \t Baseline {antenna_names[uv_max_i]} - {antenna_names[uv_max_j]},\t Distance = {np.linalg.norm(antpos[uv_max_j] - antpos[uv_max_i])}")

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

field_table = table(data_MS + "/FIELD")
print("FIELD table columns:", field_table.colnames())
print("Number of fields:", len(field_table))

# Extract source names
source_names = field_table.getcol("NAME")
print("Source names:", source_names)

# Extract source directions (RA/Dec)
source_dirs = field_table.getcol("PHASE_DIR")  # shape: (n_fields, 1, 2)
# Convert to degrees
ra_deg  = np.degrees(source_dirs[:, 0, 0])
dec_deg = np.degrees(source_dirs[:, 0, 1])
for i in range(len(source_names)):
    print(f"Field {i}: {source_names[i]} at RA = {ra_deg[i]:.6f} deg, Dec = {dec_deg[i]:.6f} deg")

field_table.close()
antennas.close()
spectral_window.close()
pol_table.close()
Pointing.close()

data = table(data_MS)

spW = np.unique(data.getcol("DATA_DESC_ID"))

time = np.unique(data.getcol("TIME"))
plt.plot(np.diff(time), ".")
plt.show()

plt.plot(data.getcol("FIELD_ID"), ".")
plt.show()

ntt = len(time)
print("DATA: ", data.colnames())
print("TIME: ", len(data.getcol("TIME")))
print("WGT : ", data.getcol("WEIGHT")[:].shape)
print("Tt  : ", ntt)
print("dt  : ", time[1]-time[0])
print("Dt  : ", (time[-1]-time[0])/3600)
print("FD1 : ", np.unique(data.getcol("FEED2")))
print("FD2 : ", np.unique(data.getcol("FEED1")))
print("FIELD : ", np.unique(data.getcol("FIELD_ID")))
print("spW : ", spW)
print("UVW : ", len(data.getcol("UVW")))

##### Check ORDER
n_bls = (nant*(nant-1))//2
print("N_BLS = ", n_bls)
check_time = data.query(query=f"DATA_DESC_ID == {1}", columns="ANTENNA1,ANTENNA2")
antenna1 = check_time.getcol("ANTENNA1")
antenna2 = check_time.getcol("ANTENNA2")
check_time.close()

# for ti in range(ntt):
#     cond1 = np.sum(antenna1[ti*n_bls:(ti+1)*n_bls]!=antenna1[:n_bls])
#     cond2 = np.sum(antenna2[ti*n_bls:(ti+1)*n_bls]!=antenna2[:n_bls])
#     if (cond1 + cond2) > 0:
#         print(ti)

check_t  = data.query(query=f"ANTENNA1 == {10} AND ANTENNA2 == {12}", columns="TIME")
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
print(np.min(d_uvw))
clean_data = []

nchan = 10
n_spW = 0

clean_data = [row['DATA'][nchan, 0] for row in data]
amplitude_crosscorrelations = np.abs(clean_data)
data.close()

plt.scatter(d_uvw, amplitude_crosscorrelations, s=1, c="k")
plt.semilogx()
plt.ylabel("Intensity")
plt.xlabel("Baseline size [$\lambda$]")
plt.show()