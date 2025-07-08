import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from   casacore.tables import table

data_FD  = "./Data/"
data_MS  = data_FD + "ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
antennas = table(data_MS+"/ANTENNA")

print("Antenna table contains ", antennas.colnames())

antenna_names = antennas.getcol("NAME")
spectral_window = table(data_MS+"/SPECTRAL_WINDOW")
print("Antenna table contains ", spectral_window.colnames())

ref_frequency = spectral_window[0]["REF_FREQUENCY"]
print("The observation was performed at frequency ", spectral_window[0]["REF_FREQUENCY"], "Hz or ",  spectral_window[0]["REF_FREQUENCY"] / 1.e6 , " MHz")
print("The number of channels available are ", len(spectral_window[0]['CHAN_FREQ']), " at frequency ", spectral_window[0]['CHAN_FREQ']/1.e6 , " MHz")

pol_table = table(data_MS + "/POLARIZATION")
corr_types = pol_table[0]["CORR_TYPE"]
print("Polarizations available:", corr_types)

Pointing = table(data_MS+"/POINTING")
Target = Pointing.getcol("TARGET")
print("Pointing ", Pointing.colnames())
print("TARGET: ", Target)
antennas.close()
spectral_window.close()
pol_table.close()
Pointing.close()

data = table(data_MS)

print("DATA: ", data.colnames())
print("TIME: ", len(data.getcol("TIME")))
print("Tt  : ", len(np.unique(data.getcol("TIME"))))
print("UVW : ", len(data.getcol("UVW")))

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

clean_data = [row['DATA'][nchan, 0] for row in data]
amplitude_crosscorrelations = np.abs(clean_data)

plt.scatter(d_uvw, amplitude_crosscorrelations, s=1, c="k")
plt.semilogx()
plt.ylabel("Intensity")
plt.xlabel("Baseline size [$\lambda$]")
plt.show()