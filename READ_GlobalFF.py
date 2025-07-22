from GlobalFF import LOAD_MS
import matplotlib.pyplot as plt
import numpy as np

# Path to the MeasurementSet (MS)
data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
# data_MS  = "../Data/n24l2.ms"

# 60, 73
ant1 = 10
ant2 = 73
npol = [0, 3]
nSpW = 0
selfbls = False
vlbi ="lofar"

ms_data = LOAD_MS(data_MS, npol=npol, SpW=nSpW, selfbls=selfbls, vlbi=vlbi)
ms_data.plot_phase_global(ant1, ant2, npol)
ms_data.close()