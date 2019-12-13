from sklearn import tree
import graphviz
import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


iris = arff.loadarff("material/iris.arff")
data = pd.DataFrame(iris[0])
data["class"] = data["class"].str.decode("UTF-8").astype("category")
data["target"] = data["class"].cat.codes
size = data.count()+1

msk = np.random.rand(len(data)) < 0.5
train = data[msk]
test = data[~msk]

column_name = data.columns[0:4]
clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(train.loc[0:,column_name], train["target"])

clf_pred = clf_train.predict(test.loc[0:,column_name])


print(confusion_matrix(clf_pred,test["target"]))
print(classification_report(clf_pred,test["target"]))
print(accuracy_score(clf_pred,test["target"]))
tree.plot_tree(clf_train)


dot_data = tree.export_graphviz(clf_train, out_file=None,
                     feature_names=column_name,
                     class_names="target",
                     filled=True, rounded=True,
                     special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("iris")
