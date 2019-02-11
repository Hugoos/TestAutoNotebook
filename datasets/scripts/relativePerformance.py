#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

def showRelativePerformanceBoxplot(scores, topList, strats, maxBaseline):
    scoreslist = []

    for score in scores:
        scoreslist.append(score['score'])
        
    topList.plot(kind="box", grid=False, figsize=(16,9), rot=45)
    y = scoreslist
    x = np.random.normal(1, 0.03, size=len(y))
    plt.plot(x, y, 'b.', alpha=0.2)
    plt.plot(np.ones(len(strats)), strats.values(), 'r.')
    plt.axhline(y=maxBaseline, color='r', linestyle='--', label=maxBaseline)
    plt.show()
