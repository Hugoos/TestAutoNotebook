#%matplotlib inline

import seaborn as sns
import openml as oml
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

from sklearn import dummy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from pymongo import MongoClient
from collections import namedtuple

from xml.parsers.expat import ExpatError

plt.rcParams['figure.dpi']= 120
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

def getDatasetTasks(did):
    tlist = oml.tasks.list_tasks()
    dtlist = {}
    for key, value in tlist.items():
        if value['did'] == did and (value['task_type'] == "Supervised Regression" or value['task_type'] == "Supervised Classification"):
            dtlist[key] = value
    return dtlist

def getTopTask(tasks):
    topAmountRuns = 0
    topTaskDict = {}
    for key, value in tasks.items():
        amountRuns = len(oml.runs.list_runs(task=[key]))
        if amountRuns > topAmountRuns:
            topAmountRuns = amountRuns
            topTaskDict = {}
            topTaskDict[key] = value
    if bool(topTaskDict): 
        for key, value in topTaskDict.items():
            return oml.runs.list_runs(task=[key])
    else:
        return -1
    
def getTopNFlows(task):
    scores = []
    strats = {}
    for id, _ in task.items():
        try:
            run = oml.runs.get_run(id)
            scores.append({"flow": run.flow_name,
                           "score": run.evaluations['area_under_roc_curve']})
        except KeyError:
            print("Flow does not have AUC evaluation.")
        except ExpatError:
            print("ExpatError, skipped run")
    scores.sort(key=operator.itemgetter('score'), reverse=True)
    try:
        strats[scores[0]["flow"]] = scores[0]["score"]

    except IndexError:
        print("No compatible runs found")
    return scores, strats

def printTopNFlows(did, N):
    tasks = getDatasetTasks(did)
    topTask = getTopTask(tasks)
    if topTask != -1:
        scores, strats = getTopNFlows(topTask)
        return topTask, pd.DataFrame.from_dict(scores)[:N], strats
    else:
        return topTask, "This dataset has no runs yet", {}
#pd.DataFrame.from_dict(tlist, orient='index')[['name','task_type','estimation_procedure']][:5]
