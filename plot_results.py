import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
import numpy as np


for n in np.linspace(0,0.04,3):
    plt.clf()
    fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(40,20))
    r = 0
    for a in np.linspace(0,1,5):
        c = 0
        for b in np.linspace(0,1,5):
            logs = []
            for e in np.linspace(0,1,5):
                l = pd.read_pickle("a=0.25_b=%s_e=%s_n=0.02_p=5_g=10"%(b,e))
                logs.append(l.rename(columns={'coherence':'e=%s'%e}))
            log = pd.concat(logs,axis=1)
            log[['e=%s'%e for e in np.linspace(0,1,5)]].plot(ax=axes[r,c],logy=True)
            log[['nr_of_states']].plot(ax=axes[r+1,c],legend=False)
            c += 1
        r += 2
    plt.savefig("results_n=%s.png"%n)