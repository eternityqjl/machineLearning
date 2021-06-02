import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

iris=sns.load_dataset("iris")

y=iris.species
X=iris.drop('species',axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100,stratify=y)

clf=tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

from sklearn.datasets import load_iris
iris=load_iris()
tree.export_graphviz(clf,out_file="iris.dot",feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,special_characters=True)

