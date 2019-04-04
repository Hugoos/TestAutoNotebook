#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import mplcursors


def showRelativePerformanceBoxplot(scores, topList, strats, maxBaseline):
    scoreslist = []
    labels = []
    fig, ax = plt.subplots()
    
    for strat in strats:
        labels.append(strat)
        
    for score in scores:
        scoreslist.append(score['score'])
        labels.append(score['flow'])
    y = scoreslist
    x = np.random.normal(1, 0.03, size=len(y))
    
    plt.plot(x, y, 'b.', alpha=0.2)
    plt.plot(np.ones(len(strats)), strats.values(), 'r.')
    plt.axhline(y=maxBaseline, color='r', linestyle='--', label=maxBaseline)
    crs = mplcursors.cursor(ax,hover=True)
    crs.connect("add", lambda sel: sel.annotation.set_text(
    '{}:\n Accuracy {}'.format(labels[sel.target.index],sel.target[1])))
    try:
        topList.plot(ax=ax, kind="box", grid=False, rot=45)
    except:
        "Cannot plot the boxplots"
        return
    fig.tight_layout()
    plt.show()
