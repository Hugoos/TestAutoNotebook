# -*- coding: utf-8 -*-

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
Runs the decision tree classification algorithm with default hyperparameters using scikit-learn DecisionTreeClassifier.

[Explanation of how a decision tree works.](http://scikit-learn.org/stable/modules/tree.html)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **criterion**:  string, optional (default=”gini”)

The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

- **max_depth**: int or None, optional (default=None)

The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- **min_samples_leaf**:  int, float, optional (default=1)

The minimum number of samples required to be at a leaf node:

- If int, then consider min_samples_leaf as the minimum number.
- If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.


- **max_features**: int, float, string or None, optional (default=None)

The number of features to consider when looking for the best split:

- If int, then consider max_features features at each split.
- If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
- If “auto”, then max_features=sqrt(n_features).
- If “sqrt”, then max_features=sqrt(n_features).
- If “log2”, then max_features=log2(n_features).
- If None, then max_features=n_features.

Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. 

- **max_leaf_nodes**: int or None, optional (default=None)

Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
"""

text_dtr = """\
Runs the decision tree regressor algorithm with default hyperparameters using scikit-learn DecisionTreeRegressor.

[Explanation of how a decision tree works.](http://scikit-learn.org/stable/modules/tree.html)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **criterion**:  string, optional (default=”mse”)

The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits, and “mae” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node.

- **max_depth**: int or None, optional (default=None)

The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- **min_samples_leaf**:  int, float, optional (default=1)

The minimum number of samples required to be at a leaf node:

- If int, then consider min_samples_leaf as the minimum number.
- If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.


- **max_features**: int, float, string or None, optional (default=None)

The number of features to consider when looking for the best split:

- If int, then consider max_features features at each split.
- If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
- If “auto”, then max_features=n_features.
- If “sqrt”, then max_features=sqrt(n_features).
- If “log2”, then max_features=log2(n_features).
- If None, then max_features=n_features.

Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

- **max_leaf_nodes**: int or None, optional (default=None)

Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
"""

text_mnb = """\
Runs the multinomial naive bayes algorithm with default hyperparameters using scikit-learn MultinomialNB.

[Explanation of how naive bayes works.](http://scikit-learn.org/stable/modules/naive_bayes.html)

The following hyperparameter has been added and can directly be changed in this notebook for further experimentation.
The description according to the sciki-learn API is:

- **alpha** : float, optional (default=1.0)

Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) """

text_rf = """\
Runs the random forest classification algorithm with default hyperparameters using scikit-learn RandomForestClassifier.

[Explanation of how a random forest works.](http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **n_estimators** : integer, optional (default=10)

The number of trees in the forest.

- **criterion**:  string, optional (default=”gini”)

The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.

- **max_depth**: integer or None, optional (default=None)

The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- **min_samples_leaf**:  int, float, optional (default=1)

The minimum number of samples required to be at a leaf node:

- If int, then consider min_samples_leaf as the minimum number.
- If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.


- **max_features**: int, float, string or None, optional (default=”auto”)

The number of features to consider when looking for the best split:

- If int, then consider max_features features at each split.
- If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
- If “auto”, then max_features=sqrt(n_features).
- If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
- If “log2”, then max_features=log2(n_features).
- If None, then max_features=n_features.

Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

- **max_leaf_nodes**: int or None, optional (default=None)

Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) """

text_svc = """\
Runs the classification support vector machine algorithm with default hyperparameters using scikit-learn SVC.

[Explanation of how a support vector machine works.](http://scikit-learn.org/stable/modules/svm.html)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **C** : float, optional (default=1.0)

Penalty parameter C of the error term.

- **kernel**:  string, optional (default=’rbf’)

Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

- **gamma**: float, optional (default=’auto’)

Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
"""

text_plot_ML = """\
Plot the accuracy of various machine learning algorithms against the baseline. """

text_lr = """\
Runs the linear regression algorithm with default hyperparameters using scikit-learn LinearRegression.

[Explanation of how linear regression works.](http://scikit-learn.org/stable/modules/linear_model.html)

[More information.](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
"""

text_rfr = """\
Runs the random forest regressor algorithm with default hyperparameters using scikit-learnRandomForestClassifierRandomForestRegressor.

[Explanation of how a random forest works.](http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **n_estimators** : integer, optional (default=10)

The number of trees in the forest.

- **criterion**:  string, optional (default=”mse”)

The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.

- **max_depth**: integer or None, optional (default=None)

The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- **min_samples_leaf**:  int, float, optional (default=1)

The minimum number of samples required to be at a leaf node:

- If int, then consider min_samples_leaf as the minimum number.
- If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.


- **max_features**: int, float, string or None, optional (default=”auto”)

The number of features to consider when looking for the best split:

- If int, then consider max_features features at each split.
- If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
- If “auto”, then max_features=n_features.
- If “sqrt”, then max_features=sqrt(n_features).
- If “log2”, then max_features=log2(n_features).
- If None, then max_features=n_features.

Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

- **max_leaf_nodes**: int or None, optional (default=None)

Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
"""

text_svr = """\
Runs the regression support vector machine algorithm with default hyperparameters using scikit-learn SVR.

[Explanation of how a support vector machine works.](http://scikit-learn.org/stable/modules/svm.html)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **C** : float, optional (default=1.0)

Penalty parameter C of the error term.

- **epsilon** : float, optional (default=0.1)

Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.

- **kernel**:  string, optional (default=’rbf’)

Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.

- **gamma**: float, optional (default=’auto’)

Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
"""

text_cknn = """\
Runs the classification k-nearest neighbours algorithm with default hyperparameters using scikit-learn KNeighborsClassifier.

[Explanation of how k-nearest neighbours works.](http://scikit-learn.org/stable/modules/neighbors.html)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **n_neighbors** : int, optional (default = 5)

Number of neighbors to use by default for kneighbors queries.

- **weights** : str or callable, optional (default = ‘uniform’)

weight function used in prediction. Possible values:

- ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
- ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
- [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

- algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional

Algorithm used to compute the nearest neighbors:

- ‘ball_tree’ will use BallTree
- ‘kd_tree’ will use KDTree
- ‘brute’ will use a brute-force search.
- ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

Note: fitting on sparse input will override the setting of this parameter, using brute force.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
"""

text_rknn = """\
Runs the regression k-nearest neighbours algorithm with default hyperparameters using scikit-learn KNeighborsRegressor.

[Explanation of how k-nearest neighbours works.](http://scikit-learn.org/stable/modules/neighbors.html)

The following hyperparameters have been added and can directly be changed in this notebook for further experimentation.
Their descriptions are according to the sciki-learn API:

- **n_neighbors** : int, optional (default = 5)

Number of neighbors to use by default for kneighbors queries.

- **weights** : str or callable

weight function used in prediction. Possible values:

- ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
- ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
- [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

Uniform weights are used by default.

- algorithm : algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional

Algorithm used to compute the nearest neighbors:

- ‘ball_tree’ will use BallTree
- ‘kd_tree’ will use KDTree
- ‘brute’ will use a brute-force search.
- ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

Note: fitting on sparse input will override the setting of this parameter, using brute force.

[More information and additional hyperparameters.](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
"""

text_plot_reg_alg = """\
Plot the accuracy of various machine learning algorithms against the baseline. """

text_clustering = """\
Clustering problems are currently not supported. """

#---------------------------------------------------------------------------------------------------------------------------


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
    fig, ax = plt.subplots()

    strats = scores
    x = np.arange(1, len(y) + 1)
    plt.plot(x, y, "o")
    plt.axhline(y=np.mean(y), color='r', linestyle='--', label='mean '+ r"$R^2$" + ' = ' + str(round(scores['mean'],4)))
    plt.axhline(y=np.median(y), color='b', linestyle='--', label='median '+ r"$R^2$" + ' = ' + str(round(scores['median'],4)))

    maxBaseline = strats[max(strats, key=strats.get)]
    
    ax.set_xlabel('Data point index')
    ax.set_ylabel(data.default_target_attribute)

    plt.legend()
    plt.show()
    return maxBaseline"""

code_plot_baseline = """\
def plot_baseline(scores):
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
    fig.autofmt_xdate()
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
dataset_name, features, importances, indices = build_forest(data)
plot_feature_importances(features, importances, indices)"""

code_baseline_plot = """\
maxBaseline = plot_baseline(baseline(data))
strats = {} """

code_regBaseline_plot = """\
scores, y = regBaseline(data)
maxBaseline = plot_regBaseline(scores, y) 
strats = {} """

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
    print("computation complexity too high, please run manually if desired.")   """

code_dtr = """\
#Runs the decision tree regressor algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
clf = tree.DecisionTreeRegressor(criterion="mse", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True);  
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
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
    print("computation complexity too high, please run manually if desired.") """

code_mnb = """\
#Runs the multinomial naive bayes algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = MultinomialNB(alpha=1.0)
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
#computational complexity O(n * p)
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
    print("computation complexity too high, please run manually if desired.") """

code_rf = """\
#Runs the random forest classifier algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below. Try changing n_trees/n_estimators and max_depth.
n_trees = 10 #Sets n_estimators such that the complexity value is calculated correctly.
clf = RandomForestClassifier(n_estimators=n_trees, criterion="gini", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
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
    print("computation complexity too high, please run manually if desired.") """

code_svc = """\
#Runs the classification support vector machine algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = svm.SVC(C =1.0, kernel="rbf", gamma="auto")
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
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
    print("computation complexity too high, please run manually if desired.") """

code_plot_ML = """\
plot_alg(strats, maxBaseline) """

code_lr = """\
#Runs the linear regression algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below. Try min_samples_leaf=5
clf = LinearRegression()
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
#computational complexity O(p^2 *n + P^3)
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
    print("computation complexity too high, please run manually if desired.") """

code_rfr = """\
#Runs the random forest algorithm on the dataset.
#Running default values, it is recommended to experiment with the values of the parameters below.
n_trees = 10 #Sets n_estimators such that the complexity value is calculated correctly.
clf = RandomForestRegressor(n_estimators=n_trees, criterion="mse", max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None)
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
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
    print("computation complexity too high, please run manually if desired.") """

code_svr = """\
#Runs the regression support vector machine algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = svm.SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="auto")
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
#computational complexity O(n^2 * p + n^3)
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
    print("computation complexity too high, please run manually if desired.") """

def code_comp(complexity):
    return """\
comp = """ + complexity

code_plot_alg_reg = """\
def plot_alg(scores, maxBaseline):
    strats = scores
    
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
    plt.xticks(range(len(strats)), list(strats.keys()))
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
    plt.show() """

code_plot_reg_alg = """\
plot_alg(strats,maxBaseline) """

code_cknn = """\
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
    print("computation complexity too high, please run manually if desired.")  """

code_rknn = """\
#Runs the regression k-nearest neighbours algorithm on the dataset
#Running default values, it is recommended to experiment with the values of the parameters below.
clf = KNeighborsRegressor(n_neighbors = 5, weights = "uniform", algorithm = "auto")
X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
folds = 10
acc = 0

X = np.nan_to_num(X) 
y = np.nan_to_num(y)

p = len(features)
n = len(X)
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
    print("computation complexity too high, please run manually if desired.") """

def code_library(dataset):
    return """\
from scripts.preamble import *
did = """ + str(dataset) + """
data = oml.datasets.get_dataset(did)
task, topList, strats = printTopNFlows(did, 10) 
topList """

run_problemType = """\
from scripts.problemType import *
problemType = findProblemType(data) 
checkTask(task, problemType, data.default_target_attribute) """

run_baselines = """\
from scripts.baselines import *
maxBaseline = generateBaselines(data, problemType) """

run_algorithms = """\
from scripts.machineLearningAlgorithms import *
strats = runMachineLearningAlgorithms(data, comp, strats, problemType, task, showRuntimePrediction=True, runTPOT=False)
plot_alg(data, strats, maxBaseline, problemType) """

run_featureImportance = """\
from scripts.featureImportance import *
featureImportance(data) """
