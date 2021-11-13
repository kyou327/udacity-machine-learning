#!/usr/bin/env python
# coding: utf-8


from __future__ import division
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np 
#sys.path.append("../tools/")
import feature_format
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', \
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', \
                     'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', \
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

poi_label = ['poi']

all_features =  poi_label + financial_features + email_features

# Determining the number of POI's
poi_count = 0
for key, value in data_dict.items():
    if value['poi']:
        poi_count+=1
total_dataset_features = []

# Validate that all features are contained in feature list and data dict
for k,v in data_dict.items():
    for i in v:
        if i not in total_dataset_features:
            total_dataset_features.append(i)

# What's python code without overused prints?
print(f'There are {len(total_dataset_features)} unique columns in the data dictionary')
print(f'There are {len(all_features)} total features in my list')

# Distribution of POI/non-POI
print(f'There are {poi_count} POIs')
print(f'There are {len(data_dict)} people in this dataset')
print('This dataset has an imbalanced y distribution')

# Let's see where those missing values are at
nans_count = {}
for i in all_features:
    nans_count[i] = 0

missing_values = {}
for k,v in data_dict.items():
    for k2, v2 in v.items(): 
        if k2 not in missing_values:
            missing_values[k2] = 0
        if v2 == 'NaN':
            missing_values[k2] += 1
            nans_count[k2]+=1
plt.bar(missing_values.keys(), missing_values.values())
plt.xticks(rotation=90)
plt.title('Missing Values (NaNs) per Feature')


features_to_remove = []
for k,v in missing_values.items():
    if v/len(data_dict) > .9:
        features_to_remove.append(k)

### Task 2: Remove outliers

### Function to plot 2 dimensions
def Plot_2dimension(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y])
    #for point in data:
        #x = point[0]
        #y = point[1]
        #plt.scatter(x, y)
    for i in data:
        salary = i[0]
        bonus = i[1]
        plt.scatter(salary, bonus)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

### Visualise outliers by 2 dimension ploting
print(Plot_2dimension(data_dict, 'salary', 'bonus'))

features_to_artificially_alter = ["salary", "bonus"]

data = feature_format.featureFormat(data_dict, features_to_artificially_alter)

for i in data:
    salary = i[0]
    bonus = i[1]
    plt.scatter(salary, bonus)
    
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
outlier = data.max()

# Find the key that is the outlier
for k,v in data_dict.items():
    if str(int(outlier)) in str(v):
        outlier_key = k
        
# The outlier is 'TOTAL'
data_dict.pop(outlier_key)

# Let's get rid of travel agency and steamrolling errors
try:
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    print('Travel agency entry removed from dictionary')
except:
    print('Travel agency not contained in dictionary')
    print('But it was just there a minute ago?')
    pass

keys_to_remove = []
for persons_name, features_dict in data_dict.items():
    nan_count = 0
    for features_dict_key, feature_value in features_dict.items():
        if feature_value == 'NaN':
            nan_count += 1
    if nan_count >= len(features_dict) * .5:
        print(f'{persons_name} is missing values for over 85% of all features')
        keys_to_remove.append(persons_name)
        

for i in keys_to_remove:
    data_dict.pop(i)
    
print(f'Will remove {features_to_remove} features')

### Create list of outliers based on dimension salary
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))
    
### Sort the list of outliers and print the top 1 outlier in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:1])

### Remove the top 1 outlier: the total line
#data_dict.pop('TOTAL', 0)

### Sort the list of outliers and print the 3 outliers in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[1:4])

### Visualise outliers by 2 dimension ploting
### lets see it again
print(Plot_2dimension(data_dict, 'salary', 'bonus'))


### Task 3: Create new feature(s)

my_dataset = data_dict

features_list = []

# iterate thru
for i in my_dataset:
    # let's see how big their bonus was compared to their salary 
    try:
        my_dataset[i]['bonus_ratio'] = float(my_dataset[i]['bonus']) / float(my_dataset[i]['salary'])
    except:
        pass
        #my_dataset[i]['bonus_ratio'] = 0.0
    # Let's see how big their expenses are compared to their pay.
    try:
        my_dataset[i]['bonus_total_ratio'] = float(my_dataset[i]['bonus']) / float(my_dataset[i]['total_payments'])
    except:
        pass
        #my_dataset[i]['bonus_total_ratio'] = 0.0
    # Let's see how many emails they sent to unsavory folk

    try:
        my_dataset[i]['emails_to_crooks_ratio'] = my_dataset[i]['from_this_person_to_poi'] / my_dataset[i]['to_messages']
    except:
        pass
        #my_dataset[i]['emails_to_crooks_ratio'] = 0.0
    # Same ratio, but different. But still the same.
    try:
        my_dataset[i]['emails_from_crooks_ratio'] = my_dataset[i]['from_messages'] / my_dataset[i]['from_poi_to_this_person']
    except:
        pass
        #my_dataset[i]['emails_from_crooks_ratio'] = 0.0
    try:
        my_dataset[i]['poi_message_percent'] = (my_dataset[i]['from_this_person_to_poi'] + my_dataset[i]['from_poi_to_this_person'])/(my_dataset[i]['to_messages']+my_dataset[i]['from_messages'])
    except:
        pass
        #my_dataset[i]['poi_message_percent'] = 0.0

new_features = ['bonus_total_ratio', 'bonus_ratio', 'emails_to_crooks_ratio','emails_from_crooks_ratio','poi_message_percent']
        
    
### Extract features and labels from dataset for "local" testing
df = pd.DataFrame.from_dict(my_dataset).T
df = df.apply(pd.to_numeric, errors='coerce')
df['poi'] = df['poi'].astype('int')
df.drop('email_address', axis=1, inplace=True)

for i in features_to_remove:
    try:
        df.drop(i, axis=1, inplace=True)
        all_features.remove(i)
    except:
        pass
    
df.fillna(0, axis=1, inplace=True)
#for i in df:
#    df.fillna(value=df[i].mean(), axis=1, inplace=True)
all_features = [x for x in df.columns]
my_dataset = df.T.to_dict()

print(f'Number of features used: {len(all_features)}')

### Task 4: Try a variety of classifiers

my_dataset = featureFormat(my_dataset, all_features, sort_keys = True)

# features is x and labels is y
# label_train, feature_train
labels_i, features_i = targetFeatureSplit(my_dataset)

### Create function for univariate feature selection with SelectKBest
def select_k_best(k):
    select_k_best = SelectKBest(k=k)
    select_k_best.fit(features_i, labels_i)
    scores = select_k_best.scores_
    unsorted_pairs = zip(all_features[1:], scores)
    k_best_features = dict(list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))[:k])
    return poi_label + list(k_best_features.keys())

### Create function to print out the features and scores by given K value
def k_best_features_score(k):
    select_k_best = SelectKBest(k=k)
    select_k_best.fit(features_i, labels_i)
    scores = select_k_best.scores_
    unsorted_pairs = zip(all_features[1:], scores)
    k_best_features = dict(list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))[:k])
    print (k_best_features)
    
### Print all features and scores with SelectKBest
print ('All features and scores:')
k_best_features_score((len(all_features)-1))

### Gaussian Naive Bayes
nb_clf = GaussianNB()
### SVC
svc_clf=SVC(probability=False)
### KNN
knn_clf = KNeighborsClassifier()
### Decision tree
dt_clf = DecisionTreeClassifier() 
### LogisticRegression
## didnt use, too many errors
#l_clf = LogisticRegression(penalty='l2')


### Evaluation metrics: Accuracy, precision, recall, f1
def evaluation(features, labels, clf, name):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train_index, test_index in cv.split(features,labels):
        features_train = [features[ii] for ii in train_index]
        features_test = [features[ii] for ii in test_index]
        labels_train = [labels[ii] for ii in train_index]
        labels_test = [labels[ii] for ii in test_index]
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)
        accuracy.append(round(accuracy_score(labels_test, labels_pred),2))
        precision.append(round(precision_score(labels_test, labels_pred, zero_division=1),2))
        recall.append(round(recall_score(labels_test, labels_pred),2))
        f1.append(f1_score(labels_test, labels_pred))
    print (name)
    print ('Mean of accuracy: {0}'.format(np.mean(accuracy)))
    print ('Mean of precision: {0}'.format(np.mean(precision)))
    print ('Mean of recall: {0}'.format(np.mean(recall)))
    print ('Mean of f1 score: {0}'.format(np.mean(f1)))
    
    
### Function: GridSearchCV to tune parameters
def find_best_params(clf, features, labels, param_grid):
    grid = GridSearchCV(clf, param_grid, cv=10)
    grid.fit(features, labels)
    return grid


### Input Parameter Grid for KNN
k_range = list(range(1,11))
algorithm_options = ['ball_tree','kd_tree','brute','auto']
param_grid_knn = dict(n_neighbors=k_range, algorithm=algorithm_options)

### Input: Parameter Grid for SVC
param_grid_svc = [
  {'C': [1, 10, 50, 100, 150, 1000], 'kernel': ['linear','rbf']},
  {'C': [1, 10, 50, 100, 150, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf']}]

### Input Parameter Grid for Decision Tree Classifier
param_grid_dt = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

### Input Parameter Grid for LogisticRegression Classfier
### decided not to use, too many errors
#param_grid_l = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


### Function to try different k value selected by SelectKBest
###run thru the classifiers to see what outputs best
def training_model(k):
    print ('k = {0}'.format(k))
    best_features = select_k_best(k)
    print(best_features)
    ### Save data_dict to my_dataset
    my_dataset = data_dict

    ### Extract features and labels from dataset
    my_dataset = featureFormat(my_dataset, best_features)
    labels, features = targetFeatureSplit(my_dataset)

    ### Scale features 
    features = MinMaxScaler().fit_transform(features)

    ### Find best params 
    knn_tune = find_best_params(knn_clf, features, labels, param_grid_knn)
    svc_tune = find_best_params(svc_clf, features, labels, param_grid_svc)
    dt_tune = find_best_params(dt_clf, features, labels, param_grid_dt)
    #l_tune = find_best_params(l_clf, features, labels, param_grid_l)

    ### Evaluation
    evaluation(features, labels, nb_clf, 'Naive Bayes Classifier (without Tuning)')
    evaluation(features, labels, knn_clf, 'K Nearest Neighbors Classifier (without Tuning)')
    evaluation(features, labels, knn_tune, 'K Nearest Neighbors Classifier (with Tuning)')
    evaluation(features, labels, svc_clf, 'SVC Classifier (without Tuning)')
    evaluation(features, labels, svc_tune, 'SVC Classifier (with Tuning)')
    evaluation(features, labels, dt_clf, 'Decision Tree Classifier (without Tuning)')
    evaluation(features, labels, dt_tune, 'Decision Tree Classifier (with Tuning)')
    #evaluation(features, labels, l_clf, 'Logistic Regression Classifier (without Tuning)')
    #evaluation(features, labels, l_tune, 'Logistic Regression Classifier (with Tuning)')
    
    
### Try different k to find out the best number of features
k_best = list(range(3,5))
for k in k_best:
    training_model(k)
    
### With k = 3, KNN classfier shows the best performance.
### Print out the best features and scores
print ('Best features selected and Scores:')
k_best_features_score(3)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

### Save the best performing classifier as clf for export
clf = knn_clf

### Save best features as features_list for export
features_list = select_k_best(4)

### Save to my_dataset for export
my_dataset = data_dict

dump_classifier_and_data(clf, my_dataset, features_list)



