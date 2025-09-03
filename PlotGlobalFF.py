"""
Baseline Phase Plotter
======================
Load a Measurement Set (MS) and plot the visibility phase
for a selected baseline.
"""

from GlobalFF import LOAD_MS

# Path to the MeasurementSet (MS)
data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
# data_MS  = "../Data/lofar-ds1-storage.ms"
# data_MS = "../Data/ILTJ123441.23+314159.4_141MHz_uv.dp3-concat"
# data_MS = "../Data/ILTJ131028.61+322045.7_143MHz_uv.dp3-concat"
# data_MS  = "../Data/n24l2.ms"

# 0, 24
ant1    = 18   
ant2    = 26
refant  = 26
npol    = [0, 3]
nSpW    = 0
selfbls = False
vlbi    = "lofar"
fld     = 0

# ant1 = 5
# ant2 = 8
# refant = 2
# npol = [0, 3]
# nSpW = 1
# selfbls = True
# vlbi = "evn"
# fld = 1

ms_data = LOAD_MS(data_MS, npol=npol, SpW=nSpW, refant=refant, selfbls=selfbls, vlbi=vlbi)
# ms_data.plot_global(ant1, ant2,"CORRECTED_DATA", savef = False, fld = 0)
ms_data.plot_phase_global(ant1, ant2, npol, Model=False, savef=False, fld=fld)

ms_data.close()
