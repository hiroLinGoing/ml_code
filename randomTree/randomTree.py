#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
titanic = pandas.read_csv("titanic_train.csv")
titanic.head(5)


# In[6]:


titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
# print(titanic["Embarked"].unique()) 可以看到s是最多 所以使用S填充
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


# In[8]:


from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#random_state就是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例
#n_estimators森林里（决策）树的数目。
#min_samples_split  分割内部节点所需要的最小样本数量
#min_samples_leaf 需要在叶子结点上的最小样本数量
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean())

alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean())

