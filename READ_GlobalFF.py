from GlobalFF import LOAD_MS

# Path to the MeasurementSet (MS)
# data_MS = "../Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"
# data_MS = "../Data/ILTJ123441.23+314159.4_141MHz_uv.dp3-concat"
data_MS = "../Data/ILTJ131028.61+322045.7_143MHz_uv.dp3-concat"
# data_MS  = "../Data/n24l2.ms"

# 0, 24
ant1 = 10
ant2 = 24
refant = 24
npol = [0, 3]
nSpW = 0
selfbls = False
vlbi = "lofar"

# ant1 = 0
# ant2 = 13
# refant = 2
# npol = [0, 3]
# nSpW = 1
# selfbls = True
# vlbi = "evn"

ms_data = LOAD_MS(data_MS, npol=npol, SpW=nSpW, refant=refant, selfbls=selfbls, vlbi=vlbi)
ms_data.plot_phase_global(ant1, ant2, npol, Model=True, savef=True)
ms_data.close()
