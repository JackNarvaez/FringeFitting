from casacore.tables     import table
import numpy as np
import matplotlib.pyplot as plt

data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

data = table(data_MS, ack=False, readonly=True)
time = data.getcol("TIME")

start_time = time.min()
end_time = start_time + 3600  # 1 hour = 3600 seconds

t1 = data.query(f"TIME >= {start_time} && TIME <= {end_time}")
data.close()

ant1 = 10
ant2 = 20
n_pol = 0

vis = t1.getcol('DATA')[:, :, n_pol]   # (channel, pol [XX, XY, YX, YX])
print(vis.shape)
flg = t1.getcol('FLAG')[:, :, n_pol]   # (channel, pol [XX, XY, YX, YX])
vis[flg] = 0.0

ant1_ =  t1.getcol('ANTENNA1')
ant2_ =  t1.getcol('ANTENNA2')
for ii in range(len(ant1_)):
    print(ant1_[ii], ant2_[ii])

phase = np.angle(vis)

plt.figure(figsize=(10, 6))
plt.imshow(phase.T, aspect='auto', cmap='seismic', vmin = -np.pi, vmax=np.pi, origin='lower')
plt.colorbar(label="Phase (radians)")
plt.ylabel("Channel")
plt.xlabel("Time index")
plt.title(f"Phase for Antennas {ant1}-{ant2}, Polarization {n_pol}")
plt.tight_layout()
plt.show()
