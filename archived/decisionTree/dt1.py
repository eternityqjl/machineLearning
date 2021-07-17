import sklearn
from sklearn import datasets,tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_feature = iris.data
iris_target = iris.target

feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33, random_state=0)

dt_model = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=3)
dt_model.fit(feature_train, target_train)
predict_results = dt_model.predict(feature_test)

print (predict_results)
print (target_test)

print (accuracy_score(predict_results, target_test))

fig=plt.figure(figsize=(10,10))
tree.plot_tree(dt_model,filled="True",
               feature_names=["Sepal length","Sepal width",
               				  "Petal length","Petal width"],
               class_names=["setosa","versicolor","virginica"])