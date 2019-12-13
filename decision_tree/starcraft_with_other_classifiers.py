from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import graphviz
import pandas as pd
from scipy.io import arff
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from matplotlib import pyplot as plt

prefix = "material/starcraft-all/"
files = os.listdir(prefix)

def obj2code(data,cla):
  data[cla] = data[cla].str.decode("UTF-8").astype("category")
  data[cla+"_target"] = data[cla].cat.codes

def main():
  # clf = tree.DecisionTreeClassifier()
  # clf = svm.SVC(gamma='scale')
  clf = MLPClassifier(solver="lbfgs")
  s_plot = pd.DataFrame(columns=['X_data', 'Y_data'])
  for file in files:
    gameTime,gameCountThreshold = re.findall(r't(\d+)_theta(\d+).arff', file)[0]

    starcraft = arff.loadarff(prefix+file)

    data = pd.DataFrame(starcraft[0])
    obj2code(data,"user")
    column_name = data.columns[3:-4]

    train_attr, test_attr, train_tar, test_tar = train_test_split(data.loc[0:,column_name],data["user_target"],test_size=0.5)

    train_clf = clf.fit(train_attr, train_tar)

    pred_clf = train_clf.predict(test_attr)

    s_plot = s_plot.append({"X_data":int(gameCountThreshold), "Y_data":accuracy_score(pred_clf,test_tar)}, ignore_index=True)

  s_plot.plot.scatter(x='X_data',y='Y_data')
  plt.show()

  # plt.title('scatter plot')
  # plt.show()


  # tree.plot_tree(clf)
  # dot_data = tree.export_graphviz(clf, out_file=None,
  #                      feature_names=column_name,
  #                      class_names="target",
  #                      filled=True, rounded=True,
  #                      special_characters=True)
  #
  # graph = graphviz.Source(dot_data)
  # graph.render("starcraft")


if __name__ == "__main__":
  main()
