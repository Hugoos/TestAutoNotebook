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

from tpot import TPOTClassifier
from tpot import TPOTRegressor

from textwrap import wrap

import openml as oml

import xml.parsers.expat

import matplotlib.pyplot as plt
import numpy as np

CLASSIFICATION = "Supervised Classification"
REGRESSION = "Supervised Regression"
CLUSTERING = "Clustering"

#High level functions

def runMachineLearningAlgorithms(data, comp, strats, problemType, task, showRuntimePrediction = False, runTPOT = False):

    if problemType == CLASSIFICATION:
        strats = decisionTreeClassifier(data, comp, strats, task, showRuntimePrediction)
        strats = naiveBayes(data, comp, strats, task, showRuntimePrediction)
        strats = randomForestClassifier(data, comp, strats, task, showRuntimePrediction)
        strats = classificationSVM(data, comp, strats, task, showRuntimePrediction)
        strats = classificationKNN(data, comp, strats, task, showRuntimePrediction)
        if runTPOT:
            strats = TPOTAutoMLClassifier(data, comp, strats, task, showRuntimePrediction)

    if problemType == REGRESSION:
        strats = decisionTreeRegressor(data, comp, strats, task, showRuntimePrediction)
        strats = linearRegression(data, comp, strats, task, showRuntimePrediction)
        strats = randomForestRegressor(data, comp, strats, task, showRuntimePrediction)
        strats = regressionSVM(data, comp, strats, task, showRuntimePrediction)
        strats = regressionKNN(data, comp, strats, task, showRuntimePrediction)
        if runTPOT:
            strats = TPOTAutoMLRegressor(data, comp, strats, task, showRuntimePrediction)
    return strats

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
    ax.set_ylim(ymin=-1)
    ax.set_ylim(ymax=1)
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

    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=1)
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

def decisionTreeClassifier(data, comp, strats, task, showRuntimePrediction):
    #Runs the decision tree classifier algorithm on the dataset
    #Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0
    
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("J48", task)

    #computational complexity O(n^2 * p)
    complexity = n**2 * p

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (n**2 * p * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['decision tree'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def naiveBayes(data, comp, strats, task, showRuntimePrediction):
    # Runs the multinomial naive bayes algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below.
    clf = MultinomialNB(alpha=1.0)
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("NaiveBayes", task)
    
    # computational complexity O(n * p)
    complexity = n * p

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if ((n * p) * x ) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['naive bayes'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def randomForestClassifier(data, comp, strats, task, showRuntimePrediction):
    # Runs the random forest classifier algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below. Try changing n_trees/n_estimators and max_depth.
    n_trees = 10 #Sets n_estimators such that the complexity value is calculated correctly.
    clf = RandomForestClassifier(n_estimators=n_trees, criterion="gini", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("RandomForest", task)

    #computational complexity O(n^2 * p * n_trees)
    complexity = n**2 * p * n_trees

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (n**2 * p * n_trees) * x > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['random forest'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def classificationSVM(data, comp, strats, task, showRuntimePrediction):
    # Runs the classification support vector machine algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below.
    clf = svm.SVC(C =1.0, kernel="rbf", gamma="auto")
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("SVM", task)

    #computational complexity O(n^2 * p + n^3)
    complexity = n**2 * p + n**3

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if ((n**2 * p + n**3) * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['support vector machine'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def classificationKNN(data, comp, strats, task, showRuntimePrediction):
        #Runs the classification k-nearest neighbours algorithm on the dataset
    #Running default values, it is recommended to experiment with the values of the parameters below.
    clf = KNeighborsClassifier(n_neighbors = 5, weights = "uniform", algorithm = "auto")
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("IBk", task)
    
    #computational complexity O(n^2 * p)
    complexity = n**2 * p * 10 #multiplying by 10 gives more accurate predictions

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (((n**2 * p)*10) * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['k-nearest neighbours'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

#Regression

def decisionTreeRegressor(data, comp, strats, task, showRuntimePrediction):
    # Runs the decision tree regressor algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
    clf = tree.DecisionTreeRegressor(criterion="mse", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);  
    folds = 10
    acc = 0

    X = np.nan_to_num(X) 
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("REPTree", task)
    
    # computational complexity O(n^2 * p)
    complexity = n**2 * p

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (n**2 * p * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds) 
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['decision tree'] = acc / folds
    else: 
        print("computation complexity too high, please run manually if desired.")
    return strats

def linearRegression(data, comp, strats, task, showRuntimePrediction):
    # Runs the linear regression algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
    clf = LinearRegression()
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("LinearRegression", task)

    # computational complexity O(p^2 *n + P^3)
    complexity = p**2 * n + p**3

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if ((p**2 * n + p**3) * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['linear regression'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def randomForestRegressor(data, comp, strats, task, showRuntimePrediction):
    # Runs the random forest algorithm on the dataset.
    # Running default values, it is recommended to experiment with the values of the parameters below.
    n_trees = 10 #Sets n_estimators such that the complexity value is calculated correctly.
    clf = RandomForestRegressor(n_estimators=n_trees, criterion="mse", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("RandomForest", task)
            
    # computational complexity O(n^2 * p * n_trees)
    complexity = n**2 * p * n_trees

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (n**2 * p * n_trees) * x > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['random forest'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def regressionSVM(data, comp, strats, task, showRuntimePrediction):
    # Runs the regression support vector machine algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below.
    clf = svm.SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="auto")
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("SMOreg", task)
            
    # computational complexity O(n^2 * p + n^3)
    complexity = n**2 * p + n**3

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (((n**2 * p) + n**3) * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['support vector machine'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
    return strats

def regressionKNN(data, comp, strats, task, showRuntimePrediction):
    # Runs the regression k-nearest neighbours algorithm on the dataset
    # Running default values, it is recommended to experiment with the values of the parameters below.
    clf = KNeighborsRegressor(n_neighbors = 5, weights = "uniform", algorithm = "auto")
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);
    folds = 10
    acc = 0

    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    p = len(features)
    n = len(X)

    if showRuntimePrediction:
        getAverageRuntime("IBk", task)
            
    # computational complexity O(n^2 * p)
    complexity = n**2 * p * 10  # multiplying by 10 gives more accurate predictions

    if complexity <= comp or comp == -1:
        for x in range(1,folds+1):
            if (((n**2 * p)*10) * x) > comp and comp != -1:
                folds = x-1
                print("Number of folds would increase the complexity over the given threshold, number of folds has been set to: " + str(folds))
                break
        if folds > len(y):
            print("Number of folds are larger than number of samples, number of folds has been set to: " + str(len(y)))
            folds = len(y)
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            acc += clf.score(X_test, y_test)
        strats['k-nearest neighbours'] = acc / folds
    else:
        print("computation complexity too high, please run manually if desired.")
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
        except ExpatError:
            print("ExpatError, skipped run")
    if count != 0:
        x = np.array(x)
        y = np.array(y)
        print("Median execution time in ms: " + str(np.median(x)))
        print("Mean execution time in ms: " + str(totalRunTime / count))
        plotScatterSimple(x,y,'usercpu_time_millis','accuracy','Execution times for ' + algName)
