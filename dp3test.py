import dp3

opr    = "flagger"
in_fl  = "./Data/ILTJ125911.17+351954.5_143MHz_uv.dp3-concat"

parset = dp3.parameterset.ParameterSet()
parset.add("average.timestep", "10")
parset.add("average.freqstep", "8")
parset.add("average.minpoints", "2")
parset.add("average.minperc", "1")

step = dp3.make_step("averager", parset, "average.", in_fl)

step_description = str(step)

print(step_description)
#step = dp3.make_step(type, parset, prefix, input_type)