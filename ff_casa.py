from casatasks     import fringefit

data_MS = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

fringefit(vis=data_MS,
          scan='30',                            # use only scan 30
          solint='inf',                         # use all timestamps in the scan
          refant='EF',                          # a big antenna does well as reference antenna
          minsnr=50,                            # empirically proven to be a good value is anything over 25
          parang=True)                          # always set to True for VLBI
