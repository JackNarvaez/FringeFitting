import numpy as np
import matplotlib.pyplot as plt
from casacore.tables     import table

# Path to the MeasurementSet (MS)
data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

# Open MS
data = table(data_MS)

antennas = table(data_MS+"/ANTENNA")
nant     = len(antennas.getcol("NAME"))
antennas.close()

valid_data = np.zeros([nant, nant])

for ant1 in range(nant):
    for ant2 in range(ant1+1, nant):
        t1 = data.query('ANTENNA1 == '+str(ant1) +' AND ANTENNA2 == '+str(ant2))   # do row selection
        flg = t1.getcol('FLAG')[:, :, :]   # (channel, pol [XX, XY, YX, YX])
        valid_data[ant1, ant2] = np.sum(flg)
data.close()
nchan, ntime, npol = flg.shape
totdata = nchan*ntime*npol
valid_data /= totdata

plt.figure(figsize=(8, 6))
plt.imshow(100*(valid_data+valid_data.T), origin='lower', vmin=0, vmax=100, cmap='jet')
plt.xlabel("Antenna 2")
plt.ylabel("Antenna 1")
plt.colorbar(label='% Flagged Data')
plt.grid(False)
plt.show()