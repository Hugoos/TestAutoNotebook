# -*- coding: utf-8 -*-
import nbformat as nbf
import warnings
import os
import sys
import openml as oml
import ast

warnings.simplefilter(action="ignore", category=RuntimeWarning)

nb = nbf.v4.new_notebook()


text_title = """\
# Automatic Jupyter Notebook for OpenML dataset"""

text_model = """\
Build Random Forest model from the dataset and compute important features. """

text_plot = """\
Plot Top-20 important features for the dataset. """

text_run = """\
Choose desired dataset and generate the most important plot. """

text_baseline = """\
Calculate baseline accuracy for classification problems using scikit-learn DummyClassifier. """

text_regBaseline = """\
Calculate baseline accuracy for regression problems using scikit-learn DummyRegressor. """

text_plot_baseline = """\
Generates a plot of the classification baseline accuracy of the various baseline strategies using scikit-learn DummyClassifier.
"""

text_plot_regBaseline = """\
Generates a plot of the regression baseline accuracy of the various baseline strategies using scikit-learn DummyRegressor.
"""

text_plot_alg = """\
Generates a plot of the accuracy of the machinelearning algorithms against the baseline.
"""

text_problemType = """\ Undetermined """

text_comp = """\
Complexity threshold to determine if an algorithm will be run.
"""

text_landmarkers = """\
The following Landmarking meta-features were calculated and stored in MongoDB: (Matthias Reif et al. 2012, Abdelmessih et al. 2010)

The accuracy values of the following simple learners are used: Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor, Decision Node, Random Node.

- **Naive Bayes Learner** is a probabilistic classifier, based on Bayes’ Theorem:
$$ p(X|Y) = \\frac{p(Y|X) \cdot p(X)}{p(Y)} $$

    where p(X) is the prior probability and p(X|Y) is the posterior probability. It is called naive, because it
    assumes independence of all attributes to each other.
- **Linear Discriminant Learner** is a type of discriminant analysis, which is understood as the grouping and separation of categories according to specific features. Linear discriminant is basically finding a linear combination of features that separates the classes best. The resulting separation model is a line, a plane, or a hyperplane, depending on the number of features combined. 

- **One Nearest Neighbor Learner** is a classifier based on instance-based learning. A test point is assigned to the class of the nearest point within the training set. 

- **Decision Node Learner** is a classifier based on the information gain of attributes. The information gain indicates how informative an attribute is with respect to the classification task using its entropy. The higher the variability of the attribute values, the higher its information gain. This learner selects the attribute with the highest information gain. Then, it creates a single node decision tree consisting of the chosen attribute as a split node. 

- **Randomly Chosen Node Learner** is a classifier that results also in a single decision node, based on a randomly chosen attribute. """

text_distances = """\
The similarity between datasets were computed based on the distance function and stored in MongoDB: (Todorovski et al. 2000)
    $$ dist(d_i, d_j) = \sum_x{\\frac{|v_{x, d_i}-v_{x, d_j}|}{max_{k \\neq i}(v_x, d_k)-min_{k \\neq i}(v_x, d_k)}}$$

where $d_i$ and $d_j$ are datasets, and $v_{x, d_i}$ is the value of meta-attribute $x$ for dataset $d_i$. The distance is divided by the range to normalize the values, so that all meta-attributes have the same range of values. """

def text_baseline_plot(ds):
    return """\
Plot of the classification baseline acuracy of the various baseline strategies using scikit-learn DummyClassifier.

The target feature is: **""" + ds.default_target_attribute + """**

The following baseline strategies are used: stratified, most_frequent, prior, uniform.

The strategies work as follow according to the sciki-learn API:

- **stratified**: Generates predictions by respecting the training set’s class distribution.

- **most_frequent**: Always predicts the most frequent label in the training set. Also known as ZeroR.

- **prior**: Always predicts the class that maximizes the class prior. 

- **uniform**: Generates predictions uniformly at random.

The horizontal red dotted line denotes the baseline value for this dataset which is equal to the best performing baseline strategy.

[More information.](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
"""

def text_regBaseline_plot(ds):
    return """\
Plot of the regression baseline acuracy of the various baseline strategies using scikit-learn DummyRegressor.

The target feature is: **""" + ds.default_target_attribute + """**

The $R^2$ statistic is calculated as a baseline.

The following baseline strategies are used: mean, median.

The strategies work as follow according to the sciki-learn API:

- **mean**: Always predicts the mean of the training set.

- **median**: Always predicts the median of the training set.

[More information.](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
"""

text_dt = """\
Runs the decision tree algorithm on the dataset WIP
"""

text_dtr = """\
Runs the decision tree regressor algorithm on the dataset WIP
"""

text_mnb = """\
Runs the multinomial naive bayes algorithm on the dataset WIP """

text_rf = """\
Runs the random forest algorithm on the dataset WIP """

text_svc = """\
Runs the classification support vector algorithm on the dataset WIP
"""

text_plot_ML = """\
Plot the accuracy of various machine learning algorithms against the baseline. """

#---------------------------------------------------------------------------------------------------------------------------



code_library = """\
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import openml as oml
import numpy as np
import pandas as pd
from sklearn import dummy
from sklearn.model_selection import train_test_split
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.dpi']= 120
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8 

from preamble import *
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from pymongo import MongoClient"""

code_baseline = """\
def baseline(data):
    strategies = ['stratified','most_frequent','prior','uniform']
    baseDict = {}
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    for strat in strategies:
        clf = dummy.DummyClassifier(strategy=strat,random_state=0)
        clf.fit(X_train, y_train)
        baseDict[strat] = clf.score(X_test, y_test)
    return baseDict  """

code_regBaseline = """\
def regBaseline(data):
    strategies = ['mean', 'median']
    baseDict = {}
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    for strat in strategies:
        clf = dummy.DummyRegressor(strategy=strat)
        clf.fit(X_train, y_train)
        baseDict[strat] = clf.score(X_test, y_test)
    return baseDict, y """

code_plot_regBaseline = """\
def plot_regBaseline(scores, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple
    
    fig, ax = plt.subplots()

    strats = scores
    x = np.arange(1, len(y) + 1)
    plt.plot(x, y, "o")
    plt.axhline(y=np.mean(y), color='r', linestyle='--', label='mean '+ r"$R^2$" + ' = ' + str(round(scores['mean'],4)))
    plt.axhline(y=np.median(y), color='b', linestyle='--', label='median '+ r"$R^2$" + ' = ' + str(round(scores['median'],4)))
    
    ax.set_xlabel('Data point index')
    ax.set_ylabel(data.default_target_attribute)

    plt.legend()
    plt.show()  """

code_plot_baseline = """\
def plot_baseline(scores):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple

    strats = scores
    maxBaseline = strats[max(strats, key=strats.get)]
    
    n_groups = len(strats)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    plt.bar(range(len(strats)), strats.values(), align='center')
    plt.xticks(range(len(strats)), list(strats.keys()))
    plt.yticks(np.arange(0, 1.1, step=0.2))
    plt.yticks(list(plt.yticks()[0]) + [maxBaseline])

    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=1)
    ax.set_xlabel('Baseline Strategy')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Performance Predicting Feature: ' + data.default_target_attribute)
    plt.axhline(y=maxBaseline, color='r', linestyle='--', label=maxBaseline)
    plt.gca().get_yticklabels()[6].set_color('red')
    fig.tight_layout()
    plt.show() 
    return maxBaseline """

code_plot_alg = """\
def plot_alg(scores, maxBaseline):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple

    strats = scores
    
    n_groups = len(strats)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    barlist =plt.bar(range(len(strats)), strats.values(), align='center')
    plt.xticks(range(len(strats)), list(strats.keys()))
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
    fig.tight_layout()
    plt.show()  """

code_forest = """\
def build_forest(data):    
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
    forest = Pipeline([('Imputer', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                       ('classifiers', RandomForestClassifier(n_estimators=100, random_state=0))])
    forest.fit(X,y)
    
    importances = forest.steps[1][1].feature_importances_
    indices = np.argsort(importances)[::-1]
    return data.name, features, importances, indices """

code_feature_plot = """\
def plot_feature_importances(features, importances, indices):
    a = 0.8
    f_sub = []
    max_features = 20

    for f in range(min(len(features), max_features)): 
            f_sub.append(f)

    # Create a figure of given size
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    # Set title
    ttl = dataset_name

    df = pd.DataFrame(importances[indices[f_sub]][::-1])
    df.plot(kind='barh', ax=ax, alpha=a, legend=False, edgecolor='w', 
            title=ttl, color = [plt.cm.viridis(np.arange(len(df))*10)])

    # Remove grid lines and plot frame
    ax.grid(False)
    ax.set_frame_on(False)

    # Customize title
    ax.set_title(ax.get_title(), fontsize=14, alpha=a, ha='left', x=0, y=1.0)
    plt.subplots_adjust(top=0.9)

    # Customize x tick lables
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.locator_params(axis='x', tight=True, nbins=5)

    # Customize y tick labels
    yticks = np.array(features)[indices[f_sub]][::-1]
    ax.set_yticklabels(yticks, fontsize=8, alpha=a)
    ax.yaxis.set_tick_params(pad=2)
    ax.yaxis.set_ticks_position('none')  
    ax.set_ylim(ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5) 

    # Set x axis text
    xlab = 'Feature importance'
    ax.set_xlabel(xlab, fontsize=10, alpha=a)
    ax.xaxis.set_label_coords(0.5, -0.1)

    # Set y axis text
    ylab = 'Feature'
    ax.set_ylabel(ylab, fontsize=10, alpha=a)
    plt.show() """


code_get_landmarkers = """\
def connet_mongoclient(host):
    client = MongoClient('localhost', 27017)
    db = client.landmarkers
    return db
    
def get_landmarkers_from_db():
    db = connet_mongoclient('109.238.10.185')
    collection = db.landmarkers2
    df = pd.DataFrame(list(collection.find()))
    
    landmarkers = pd.DataFrame(df['score'].values.tolist())
    datasetID = df['dataset'].astype(int)
    datasets = oml.datasets.get_datasets(datasetID)
    return df, landmarkers, datasetID, datasets """

code_get_distances = """\
def get_distance_from_db():
    db = connet_mongoclient('109.238.10.185')
    collection = db.distance
    df = pd.DataFrame(list(collection.find()))
    distance = list(df['distance'].values.flatten())
    return distance """


code_compute_similar_datasets = """\
def compute_similar_datasets(dataset):
    n = 30
    dataset_index = df.index[datasetID == dataset][0]
    similar_dist = distance[:][dataset_index]
    similar_index = np.argsort(similar_dist)[:n]
    return similar_index """

code_get_datasets_name = """\
def get_datasets_name(datasets, similar_index):
    n = 30
    datasets_name = []

    for i in similar_index:
        datasets_name.append(datasets[i].name)    
    return datasets_name """

code_run = """\
data = oml.datasets.get_dataset(dataset)
dataset_name, features, importances, indices = build_forest(data)
plot_feature_importances(features, importances, indices)"""

code_baseline_plot = """\
maxBaseline = plot_baseline(baseline(data))  """

code_regBaseline_plot = """\
scores, y = regBaseline(data)
plot_regBaseline(scores, y) """

code_landmarkers_plot = """\
sns.set(style="whitegrid", font_scale=0.75)
f, ax = plt.subplots(figsize=(8, 4))

df, landmarkers, datasetID, datasets = get_landmarkers_from_db()
landmarkers.columns = ['One-Nearest Neighbor', 'Linear Discriminant Analysis', 'Gaussian Naive Bayes', 
                       'Decision Node', 'Random Node']

distance = np.squeeze(get_distance_from_db())
similar_index = compute_similar_datasets(dataset)
sns.violinplot(data=landmarkers.iloc[similar_index], palette="Set3", bw=.2, cut=1, linewidth=1)
sns.despine(left=True, bottom=True) """

code_similarity_plot = """\
datasets_name = get_datasets_name(datasets, similar_index)
sns.set(style="white")
corr = pd.DataFrame(distance[similar_index[:, None], similar_index], 
                    index = datasets_name, columns = datasets_name)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, mask=mask, cmap = "BuPu_r", vmax= 1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})"""

code_dt_1 = """\
#Runs the decision tree algorithm on the dataset
from sklearn import tree
#Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

p = len(features)
n = len(X_train)
#computational complexity O(n^2 * p)
complexity = n**2 * p

if complexity <= comp or comp == -1:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    strats = baseline(data)
    strats['Decision Tree'] = acc 
else: 
    print("computation complexity too high, please run manually if desired.") """

code_dtr = """\
#Runs the decision tree regressor algorithm on the dataset
from sklearn import tree
#Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
clf = tree.DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)
y_train = np.nan_to_num(y_train)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
#strats = baseline(data)
#strats['Decision Tree'] = acc
print(acc) """

code_mnb = """\
#Runs the Multinomial Naive Bayes algorithm on the dataset
from sklearn.naive_bayes import MultinomialNB
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = MultinomialNB()
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

p = len(features)
n = len(X_train)
#computational complexity O(n * p)
complexity = n * p

if complexity <= comp or comp == -1:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    strats['naive bayes'] = acc 
else: 
    print("computation complexity too high, please run manually if desired.") """

code_rf = """\
#Runs the Random Forest algorithm on the dataset
from sklearn.ensemble import RandomForestClassifier
import math
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = RandomForestClassifier()
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

p = len(features)
n = len(X_train)
#computational complexity O(n^2 * sqrt(p * n_trees))
complexity = n**2 * math.sqrt(p * 10)

if complexity <= comp or comp == -1:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    strats['random forest'] = acc 
else: 
    print("computation complexity too high, please run manually if desired.") """

code_svc = """\
#Runs the Classification Support Vector Machine algorithm on the dataset
from sklearn import svm
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = svm.SVC()
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

p = len(features)
n = len(X_train)
#computational complexity O(n^2 * p + n^3)
complexity = n**2 * p + n**3

if complexity <= comp or comp == -1:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    strats['support vector machine'] = acc 
else: 
    print("computation complexity too high, please run manually if desired.") """

code_plot_ML = """\
plot_alg(strats, maxBaseline) """

def code_comp(complexity):
    return """\
comp = """ + complexity 



def main():
    # print command line arguments sys.argv[1:]:
    datasets = ast.literal_eval(sys.argv[1])
    if len(sys.argv) > 2:
        comp = sys.argv[2]
    else:
        comp = "default"
        
    complexity = calc_comp(comp)

        
    if isinstance(datasets, list):
        for dataset in datasets:
            print("Generating jupyter notebook for dataset "+str(dataset)+"...")
            generate_jnb(dataset, complexity)
    else:
        print("Generating jupyter notebook for dataset "+str(datasets)+"...")
        generate_jnb(datasets, complexity)
        
def calc_comp(comp):
    comp = comp.lower()
    if comp == "low":
        return 5000000000000
    elif comp == "mid":
        return 50000000000000
    elif comp == "high":
        return 5000000000000000
    elif comp == "inf":
        return -1
    else:
        return 50000000000000

def create_block(text, code):
    if isinstance(text, list):
        for t in text:
            nb['cells'].append(nbf.v4.new_markdown_cell(t))
    else:
        nb['cells'].append(nbf.v4.new_markdown_cell(text))
    if isinstance(code, list):
        for c in code:
            nb['cells'].append(nbf.v4.new_code_cell(c))
    else:
        nb['cells'].append(nbf.v4.new_code_cell(code))



def isRegressionProblem(data):
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True)
    total = 0
    global text_problemType
    uniqueElementsDict = {}
    answer = -1
    for item in y:
        total += 1
        uniqueElementsDict[item] = ""
    
    perUnique = len(uniqueElementsDict) / total
    
    if perUnique > 0.05:
        text_problemType = """\
The percentage of unique values for the default target attribute in this data set is """ + str(round(perUnique,4)) + """.
Because this is higher than 5% of the dataset we assume that this is a **regression** problem. """
        answer = 1
    else:
        text_problemType = """\
The percentage of unique values for the default target attribute in this data set is """ + str(round(perUnique,4)) + """.
Because this is lower or equal than 5% of the dataset we assume that this is a **classification** problem. """
        answer = 0

    return answer

        


def generate_jnb(dataset, complexity):
    nb['cells'] = []
    fname = str(dataset)+'.ipynb'
    ds = oml.datasets.get_dataset(dataset)
    isRegression = isRegressionProblem(ds)
    text_title = """\
    # Automatic Jupyter Notebook for OpenML dataset %s: %s""" % (dataset,ds.name)
    create_block(text_title, code_library)
    nb['cells'].append(nbf.v4.new_markdown_cell(text_problemType))
    create_block(text_comp, code_comp(str(complexity)))
    if not isRegression:
        create_block(text_baseline, code_baseline)
    if isRegression:
        create_block(text_regBaseline, code_regBaseline)
    if not isRegression:
        create_block(text_plot_baseline, code_plot_baseline)
    if isRegression:
        create_block(text_plot_regBaseline, code_plot_regBaseline)
    create_block(text_plot_alg, code_plot_alg)
    create_block(text_model, code_forest)
    create_block(text_plot, code_feature_plot)
    create_block(text_run, ["dataset = " + str(dataset), code_run])
    if not isRegression:
        create_block(text_baseline_plot(ds), code_baseline_plot)
    if isRegression:
        create_block(text_regBaseline_plot(ds), code_regBaseline_plot)
    if not isRegression:
        #All classification algorithms here
        create_block(text_dt,code_dt_1)
        create_block(text_mnb,code_mnb)
        create_block(text_rf,code_rf)
        create_block(text_svc,code_svc)
        create_block(text_plot_ML,code_plot_ML)
    if isRegression:
        #All regression algorithms here
        create_block(text_dtr,code_dtr)
        
    
    #create_block(text_landmarkers,code_get_landmarkers)
    #create_block(text_distances,[code_get_distances,code_compute_similar_datasets,code_get_datasets_name,code_landmarkers_plot,code_similarity_plot])
    
    # nb['cells'] = [nbf.v4.new_markdown_cell(text_title),
    #  nbf.v4.new_code_cell(code_library),
    # nbf.v4.new_markdown_cell(text_model),
    # nbf.v4.new_code_cell(code_forest),
    # nbf.v4.new_markdown_cell(text_plot),
    # nbf.v4.new_code_cell(code_feature_plot),
    # nbf.v4.new_markdown_cell(text_run),
    # nbf.v4.new_code_cell("dataset ="+ str(dataset)),
    # nbf.v4.new_code_cell(code_run),
    # nbf.v4.new_markdown_cell(text_landmarkers),
    # nbf.v4.new_code_cell(code_get_landmarkers),
    # nbf.v4.new_markdown_cell(text_distances),
    # nbf.v4.new_code_cell(code_get_distances),
    # nbf.v4.new_code_cell(code_compute_similar_datasets),
    # nbf.v4.new_code_cell(code_get_datasets_name),
    # nbf.v4.new_code_cell(code_landmarkers_plot),
    # nbf.v4.new_code_cell(code_similarity_plot)]

    with open(fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    os.system("jupyter nbconvert --execute --inplace %s"%(fname))
    

if __name__ == "__main__":
    main()

