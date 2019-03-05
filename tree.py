import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

from sklearn.datasets.california_housing import fetch_california_housing
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
decisionTree = tree.DecisionTreeRegressor(random_state = 42)
#housing.data[:, [0, 1]] 输入样本x  housing.target样本标签
decisionTree.fit(data_train, target_train)
decisionTree.score(data_test, target_test)

from sklearn.grid_search import GridSearchCV
# criterion='mse', max_depth=None, max_features=None,
#            max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=1,
#            min_samples_split=3, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best'
tree_param_grid = { 'min_samples_split': list((3,6,9)),'max_depth':list((2,4,6,8,16, 32))}
grid = GridSearchCV(tree.DecisionTreeRegressor(),param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
grid.grid_scores_, grid.best_params_, grid.best_score_
