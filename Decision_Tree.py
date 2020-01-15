import pandas as pd 
import numpy
import graphviz 
from sklearn import preprocessing
from sklearn import tree 
from sklearn.model_selection import train_test_split


data_file = "data/mammographic_masses.data.txt"
column_names = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
data = pd.read_csv(data_file, na_values=['?'], names = column_names) # some cell in the data file is missing, convert the missing data to NaN

drop_rows = [(data['BI-RADS'].isnull()) | (data['Age'].isnull()) | (data['Shape'].isnull()) | (data['Margin'].isnull()) | (data['Density'].isnull())] # the rows that contains NaN if the row index refers to is true, needs to be dropped

data.dropna(inplace=True)

train_attributes = data[['Age', 'Shape', 'Margin', 'Density']].values # we are using these four attributes, BI-RADS is not predictive so we don't use it

predict_attributes = data['Severity'].values # we want to predict the value in Serverity column

attributes_names = ['Age', 'Shape', 'Margin', 'Density']

scaler = preprocessing.StandardScaler()
train_attributes_scaled = scaler.fit_transform(train_attributes) # normalize data for better performance

numpy.random.seed(567)

(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(train_attributes_scaled, predict_attributes, train_size=0.8, random_state=1) # 80% data will be trained and 20% will be used as a testing data

clf = tree.DecisionTreeClassifier(random_state=1)

clf.fit(training_inputs, training_classes) # train the classifier

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=attributes_names)
graph = graphviz.Source(dot_data) 
graph.render("tree",view = True) # display the decision tree as tree.pdf

accuracy = clf.score(testing_inputs, testing_classes) # the portion of correctly predicted data in testing set 

print("Decision Tree Accuracy: " + str(accuracy))