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
from sklearn import impute
from sklearn import *

from tpot import TPOTClassifier
from tpot import TPOTRegressor

from textwrap import wrap

import openml as oml
from openml import tasks, flows, runs
from openml.exceptions import PyOpenMLError

from xml.parsers.expat import ExpatError

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from itertools import combinations

CLASSIFICATION = "Supervised Classification"
REGRESSION = "Supervised Regression"
CLUSTERING = "Clustering"

#High level functions

def runMachineLearningAlgorithms(data, comp, strats, problemType, task, showRuntimePrediction = False, runTPOT = False, timeLimit = 300000):
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    p = len(features)
    n = len(X)
    #unitValueMs = 0.01135917705 
    if problemType == CLASSIFICATION:
        strats = runMLAlgorithm(tree.DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None),
                                "decision tree", strats, task, showRuntimePrediction, "J48", timeLimit, isTooLong(n**2 * p, comp))
        strats = runMLAlgorithm(MultinomialNB(alpha=1.0),
                                "naive bayes", strats, task, showRuntimePrediction, "NaiveBayes", timeLimit, isTooLong(n * p, comp))
        strats = runMLAlgorithm(RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None),
                                "random forest", strats, task, showRuntimePrediction, "RandomForest", timeLimit, isTooLong(n**2 * p * 10, comp))
        strats = runMLAlgorithm(svm.SVC(C =1.0, kernel="rbf", gamma="auto"),
                                "support vector machine", strats, task, showRuntimePrediction, "SVM", timeLimit, isTooLong(n**2 * p + n**3, comp))
        strats = runMLAlgorithm(KNeighborsClassifier(n_neighbors = 5, weights = "uniform", algorithm = "auto"),
                                "k-nearest neighbours", strats, task, showRuntimePrediction, "IBk", timeLimit, isTooLong(n**2 * p * 10, comp)) #multiplying by 10 gives more accurate predictions
        if runTPOT:
            strats = TPOTAutoMLClassifier(data, comp, strats, task, showRuntimePrediction)

    if problemType == REGRESSION:
        strats = runMLAlgorithm(tree.DecisionTreeRegressor(criterion="mse", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None),
                                "decision tree", strats, task, showRuntimePrediction, "REPTree", timeLimit, isTooLong(n**2 * p, comp))
        strats = runMLAlgorithm(LinearRegression(),
                                "linear regression", strats, task, showRuntimePrediction, "LinearRegression", timeLimit, isTooLong(p**2 * n + p**3, comp))
        strats = runMLAlgorithm(RandomForestRegressor(n_estimators=10, criterion="mse", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None),
                                "random forest", strats, task, showRuntimePrediction, "RandomForest", timeLimit, isTooLong(n**2 * p * 10, comp))
        strats = runMLAlgorithm(svm.SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="auto"),
                                "support vector machine", strats, task, showRuntimePrediction, "SMOreg", timeLimit, isTooLong(n**2 * p + n**3, comp))
        strats = runMLAlgorithm(KNeighborsRegressor(n_neighbors = 5, weights = "uniform", algorithm = "auto"),
                                "k-nearest neighbours", strats, task, showRuntimePrediction, "IBk", timeLimit, isTooLong(n**2 * p * 10, comp)) # multiplying by 10 gives more accurate predictions
        if runTPOT:
            strats = TPOTAutoMLRegressor(data, comp, strats, task, showRuntimePrediction)
    return strats

def algorithmText(strats, maxBaseline):
    text = ""
    badPerformance = False
    badPerformingAlgorithms = []

    #Text about algorithms worse than baseline
    for key in strats.keys():
        if strats[key] < maxBaseline:
            badPerformance = True
            badPerformingAlgorithms.append(key)
    if badPerformance:
        if len(badPerformingAlgorithms) < 2:
            text += "One of the algorithms: " + str(badPerformingAlgorithms)[1:-1] + " performed worse than the baseline.\n"
        else:
            text += "Several of the algorithms: " + str(badPerformingAlgorithms)[1:-1] + " performed worse than the baseline.\n"
    else:
        text += "All algorithms performed better than the baseline.\n"

    #Text about algorithm performance
    for key in strats.keys():
        text += str(key) + " has an accuracy of " + str(strats[key]) + ".\n"    
    print(text)

def isTooLong(runtime, comp):
    return (runtime > comp)

def plot_alg(data, strats, maxBaseline, problemType):

    if problemType == CLASSIFICATION:
        plot_class_alg(data, strats, maxBaseline)

    if problemType == REGRESSION:
        plot_reg_alg(data, strats, maxBaseline)

#Plotting

def plot_reg_alg(data, strats, maxBaseline):
    n_groups = len(strats)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    x = list(strats.keys())
    y = []
    for strat in x:
        y.append(strats[strat])
    markerline, stemlines, baseline = plt.stem(np.arange(n_groups), y, '-.')
    plt.setp(baseline, color='r', linewidth=2)
    stratsList = list(strats.keys())
    plt.xticks(range(len(strats)), [ '\n'.join(wrap(l, 20)) for l in stratsList])
    plt.yticks(np.arange(-1, 1.1, step=0.2))
    plt.yticks(list(plt.yticks()[0]) + [maxBaseline])
    plt.plot()
    ax.set_ylim(bottom=-1)
    ax.set_ylim(top=1)
    ax.set_xlim(xmin=-0.1)
    ax.set_xlim(xmax=len(strats)-0.9)
    ax.set_xlabel('Machine Learning Algorithm')
    ax.set_ylabel('$R^2$')
    ax.set_title('Algorithm Performance Predicting Feature: ' + data.default_target_attribute)
    plt.axhline(y=maxBaseline, color='r', linestyle='--', label=maxBaseline)
    plt.gca().get_yticklabels()[len(plt.gca().get_yticklabels())-1].set_color('red')
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def plot_class_alg(data, strats, maxBaseline):
    n_groups = len(strats)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    barlist =plt.bar(range(len(strats)), strats.values(), align='center')
    stratsList = list(strats.keys())
    plt.xticks(range(len(strats)), [ '\n'.join(wrap(l, 20)) for l in stratsList])
    plt.yticks(np.arange(0, 1.1, step=0.2))
    plt.yticks(list(plt.yticks()[0]) + [maxBaseline])

    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1)
    ax.set_xlabel('Machine Learning Algorithm')
    ax.set_ylabel('Accuracy')
    ax.set_title('Algorithm Performance Predicting Feature: ' + data.default_target_attribute)
    plt.axhline(y=maxBaseline, color='r', linestyle='--', label=maxBaseline)
    plt.gca().get_yticklabels()[6].set_color('red')
    for bar in barlist:
        if bar.get_height() > maxBaseline:
            bar.set_facecolor('g')
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def plotScatterSimple(x,y,xlabel,ylabel,title):

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.grid()
    
    plt.show()

#Classification

def outlierDetection(X, features): 
    clf = pipeline.Pipeline(
            steps=[
                ('imputer', impute.SimpleImputer()),
                ('estimator', ensemble.IsolationForest())
            ]
        )
    outliers = clf.fit_predict(X)
    df = pd.DataFrame(X, columns=features)
    plotCombinations = combinations(df.keys(), 2)
    df['outlier'] = outliers    
    for a,b in plotCombinations:
        fig, ax = plt.subplots() #required due to bug in pandas see https://github.com/jupyter/notebook/issues/2353
        df.plot.scatter(x=a,y=b,c='outlier',colormap='bwr_r', ax=ax)
        #plt.tight_layout()
    
    

def runMLAlgorithm(estimator, name, strats, task, showRuntimePrediction, RTPName, timeLimit, tooLong):
    acc = 0
    expectedRuntime = -1
    if showRuntimePrediction:
        expectedRuntime = getAverageRuntime(RTPName, task)
    if (expectedRuntime <= timeLimit and expectedRuntime != -1) or (not tooLong and expectedRuntime  == -1):
        taskId = tasks.get_task(getTaskId(task))
        
        clf = pipeline.Pipeline(
            steps=[
                ('imputer', impute.SimpleImputer()),
                ('estimator', estimator)
            ]
        )
        
        flow = flows.sklearn_to_flow(clf)
        try:
            run = runs.run_flow_on_task(taskId, flow, avoid_duplicate_runs = True)
        except PyOpenMLError:
            print("Run already exists in OpenML, WIP")
            return strats
        feval = dict(run.fold_evaluations['predictive_accuracy'][0])

        for val in feval.values():
            acc += val
        strats[name] = acc / 10
        run.publish()
        run.push_tag("auto-jupyter-notebook")
    else:
        print("Skipping run because of time limit set")
    return strats

def TPOTAutoMLClassifier(data, comp, strats, task, showRuntimePrediction):
    # Runs the AutoML algorithm on the dataset
    clf = TPOTClassifier()
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    #if showRuntimePrediction:
    #    getAverageRuntime("IBk", task)
            
    # computational complexity O(n^2 * p)
    #complexity = n**2 * p * 10 

    #if complexity <= comp or comp == -1:
    #for x in range(1,folds+1):
    #    if (((n**2 * p)*10) * x) > comp and comp != -1:
    #        folds = x-1
    #        print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
    #        break
    #if folds > len(y):
    #    print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
    #    folds = len(y)
    
    #kf = KFold(n_splits=folds)
    #for train_index, test_index in kf.split(X,y):
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    strats['TPOTAutoML'] = acc 

def TPOTAutoMLRegressor(data, comp, strats, task, showRuntimePrediction):
    # Runs the AutoML algorithm on the dataset
    clf = TPOTRegressor()
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    #if showRuntimePrediction:
    #    getAverageRuntime("IBk", task)
            
    # computational complexity O(n^2 * p)
    #complexity = n**2 * p * 10 

    #if complexity <= comp or comp == -1:
    #for x in range(1,folds+1):
    #    if (((n**2 * p)*10) * x) > comp and comp != -1:
    #        folds = x-1
    #        print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
    #        break
    #if folds > len(y):
    #    print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
    #    folds = len(y)
    #kf = KFold(n_splits=folds)
    #for train_index, test_index in kf.split(X,y):
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    strats['TPOTAutoML'] = acc 
    #else:
    #    print("computation complexity too high, please run manually if desired.")
    return strats

# Runtime approximation

def getAverageRuntime(algName, task):
    if task == -1:
        return "No prior runtime known"
    count = 0
    totalRunTime = 0
    x = []
    y = []
    for id, _ in task.items():
        try:
            run = oml.runs.get_run(id)
            if algName in run.flow_name:
                if 'usercpu_time_millis' in run.evaluations and 'predictive_accuracy' in run.evaluations:
                    count += 1
                    x.append(run.evaluations['usercpu_time_millis'])
                    y.append(run.evaluations['predictive_accuracy'])
                    totalRunTime += run.evaluations['usercpu_time_millis']
        except KeyError:
            print("KeyError")
            return -1
        except ExpatError:
            print("ExpatError, skipped run")
            return -1
    if count != 0:
        x = np.array(x)
        y = np.array(y)
        print("Median execution time in ms: " + str(np.median(x)))
        print("Mean execution time in ms: " + str(totalRunTime / count))
        plotScatterSimple(x,y,'usercpu_time_millis','accuracy','Execution times for ' + algName)
        return np.median(x)
    return -1

def getTaskId(task):
    return task[next(iter(task))]['task_id']

def testFunction(data):
    #clf = sklearn.ensemble.forest.RandomForestClassifier(bootstrap:true,weight:null,criterion:"gini",depth:null,features:"auto",nodes:null,decrease:0.0,split:null,leaf:1,split:2,leaf:0.0,estimators:10,jobs:1,score:false,state:6826,verbose:0,start:false)
    #X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);

    run = oml.runs.get_run(1836360)
    print(run.flow_id)
    #flow = oml.flows.get_flow(4834)
    flow = oml.flows.get_flow(8900)
    #flow = oml.flows.get_flow(8426)
    #flow = oml.flows.get_flow(7650)
    flow = oml.flows.flow_to_sklearn(flow)
    clf = pipeline.Pipeline(
            steps=[
                ('imputer', impute.SimpleImputer()),
                ('estimator', flow)
            ]
        )
    flow = flows.sklearn_to_flow(clf)
    print(flow.model)
    taskId = tasks.get_task(55)

    
    run = runs.run_flow_on_task(taskId, flow, avoid_duplicate_runs = True)

    feval = dict(run.fold_evaluations['predictive_accuracy'][0])
    acc = 0
    for val in feval.values():
        acc += val
    print(acc / 10)
    #X = np.nan_to_num(X)
    #y = np.nan_to_num(y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    #clf.fit(X_train, y_train)
    #acc = clf.score(X_test, y_test)
    #print(acc)
