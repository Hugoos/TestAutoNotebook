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
from sklearn import impute
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest

from pymongo import MongoClient
from collections import namedtuple

from xml.parsers.expat import ExpatError

def highlight_1D_Outliers(s):
    '''
    highlight the outliers in a Series yellow.
    '''
    is_outlier = s == s.quantile(0.99)
    return ['background-color: yellow' if v else '' for v in is_outlier]

def outlierDetection(X, features, N): 
    clf = Pipeline(
            steps=[
                ('imputer', impute.SimpleImputer()),
                ('estimator', IsolationForest(behaviour='new', contamination='auto'))
            ]
        )
    clf.fit(X)
    outliers = clf.decision_function(X)
    df = pd.DataFrame(X, columns=features)
    originalFeatures = df.keys()
    normalized_df=(df-df.mean())/df.std()
    normalized_df.plot(kind="box", grid=False, figsize=(16,9), rot=45)
    #plotCombinations = combinations(df.keys(), 2)
    dfo = pd.DataFrame({"outlier": outliers})
    df = df.join(dfo)
    df = df.sort_values(by=['outlier'])
    cm = sns.light_palette("red", as_cmap=True, reverse=True)
    return(df[:N].style.\
            background_gradient(subset=['outlier'], cmap=cm).\
            apply(subset=originalFeatures, func=highlight_1D_Outliers))
    #for a,b in plotCombinations:
    #    fig, ax = plt.subplots() #required due to bug in pandas see https://github.com/jupyter/notebook/issues/2353
    #    df.plot.scatter(x=a,y=b,c='outlier',colormap='bwr_r', ax=ax)
        #plt.tight_layout()
