import numpy as np
import matplotlib.pyplot as plt

ant1 = 30
dant = 3
delay_max = np.load("delay_baseline_"+str(ant1)+"_"+str(dant)+".npy")
rate_max  = np.load("rate_baseline_"+str(ant1)+"_"+str(dant)+".npy")

nt, nd, _ = delay_max.shape

cc = [
    '#e6194b', '#3cb44b', '#ffe119', '#0082c8',
    '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#d2f53c', '#fabebe', '#008080', '#e6beff',
    '#aa6e28', '#fffac8', '#800000', '#aaffc3'
]

fig, axs = plt.subplots(2,nd, figsize=(10*nd, 6), sharex=True)
plt.subplots_adjust(hspace=0.05)
plt.subplots_adjust(wspace=0.3)
fig.suptitle(f"Ant1 = {ant1}", fontsize=16)
antt = range(ant1+dant, 74, dant)#np.arange(ant1+dant, 74, dant)

print(nt, nd, len(antt))
for ii in range(nd):

    #delay
    pp = int(2**ii)
    delaybl = delay_max[:, ii, :pp]
    ratebl  = rate_max[:, ii, :pp]

    for jj in range(pp):
        axs[0, ii].scatter(antt, delaybl[:, jj], s=5, color=cc[jj])
        axs[1, ii].scatter(antt, ratebl[:, jj],  s=5, color=cc[jj])
    
    axs[1, ii].set_xlabel("Ant2", fontsize=15)
    axs[0, ii].set_xticks(antt[::2])

axs[0, 0].set_ylabel("delay [ns]", fontsize=15)
axs[1, 0].set_ylabel("fringe rate [mHz]", fontsize=15)
plt.show()
