from sklearn import tree
import pandas as pd
from scipy.io import arff
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import statistics

prefix = "starcraft-all/"
files = os.listdir(prefix)
plot_colors = "ryb"
plot_step = 0.02



def obj2code(data, cla):
    data[cla] =  data[cla].str.decode("UTF-8").astype("category")
    data[cla + "_target"] =  data[cla].cat.codes

def decision_tree(name):
    ##clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate='adaptive', hidden_layer_sizes=(5, 2), random_state=1)
    s_plot = pd.DataFrame(columns=['X_data', 'Y_data', 'Z_data'])
    clf = tree.DecisionTreeClassifier()
    ##clf = svm.SVC(gamma='auto')

    for file in files:
        gameTime, gameCountThreshold = re.findall(r't(\d+)_theta(\d+).arff', file)[0]

        load = arff.loadarff(prefix + file)
        data = pd.DataFrame(load[0])
        obj2code(data, name)
        column_name = data.columns[3:-4]
        train_attr, test_attr, train_tar, test_tar = train_test_split(data.loc[0:,column_name],data[name +"_target"],test_size=0.5)


        train_clf = clf.fit(train_attr, train_tar)

        pred_clf = train_clf.predict(test_attr)
        acc_score = accuracy_score(test_tar, pred_clf)
        depth = train_clf.get_depth()
        s_plot = s_plot.append({"X_data":int(gameCountThreshold),'Y_data': acc_score, 'Z_data': depth}, ignore_index=True)

    return s_plot



def main():
    cla = ["user", "status", "race"]
    threedee = plt.figure().gca(projection='3d')
    #for one pred only s_plot = decision_tree("user")
    # for one pred only threedee.scatter(s_plot['Z_data'], s_plot['X_data'], s_plot['Y_data'], c="r", cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
    for class_name, color in zip(cla, plot_colors):
        s_plot = decision_tree(class_name)
        threedee.scatter(s_plot['Z_data'], s_plot['X_data'], s_plot['Y_data'], c=color, label=class_name, cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
    threedee.scatter(s_plot['Z_data'], s_plot['X_data'], s_plot['Y_data'])
    threedee.set_xlabel('Tree Depth')
    threedee.set_ylabel('Threshold')
    threedee.set_zlabel('Accuracy')
    plt.show()


if __name__ == "__main__":
  main()

##guillaume.bono@insa-lyon.fr
##
