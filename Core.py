parset_str = \
"""
msin=ILTJ123441.23+314159.4_141MHz_uv.dp3-concat 
steps=[add,filter] 
msout=ILTJ123441.23+314159.4_141MHz_uv.dp3-concatsc
msin.datacolumn=DATA
filter.type=filter 
filter.remove=True 
msout.uvwcompression=False 
msout.storagemanager=dysco 
msout.storagemanager.weightbitrate=16 
add.type=stationadder 
add.stations={ST001:'CS*'} 
filter.baseline='!CS*&&*' 
"""

with open("apply.parset", "w") as f_out:
    f_out.write(parset_str)