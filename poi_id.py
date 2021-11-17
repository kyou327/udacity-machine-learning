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

## Let's get rid of travel agency and steamrolling errors
## added removal of Lockhart Eugene E for NaNs on second commit
try:
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    print('Travel agency entry removed from dictionary')
except:
    print('Travel agency not contained in dictionary')
    print('But it was just there a minute ago?')
    pass
try:
    data_dict.pop('LOCKHART EUGENE E')
    print('LOCKHART EUGENE E removed from dictionary')
except:
    print('LOCKHART EUGENE E not contained in dictionary')
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
    

### Sort the list of outliers and print the top 3 outliers in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[0:3])

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

### Task 4: Try a variety of classifiers

data = featureFormat(my_dataset, all_features, sort_keys = True)

# features is x and labels is y
# label_train, feature_train
labels, features = targetFeatureSplit(data)

f_values = f_classif(features, labels)

significant_count = 0
significant_idx = []
print('P-values:')
for i in range(f_values[0].shape[0]):
    if f_values[1][i] < .05:
        significant_count += 1
        significant_idx.append((i, f_values[1][i]))
#print(f'Significant Features: {str(significant_idx).replace("[","").replace("]","")} ')
print(f'There are {significant_count} significant features')
print('I will use these 17 most significant features in my analysis')

significant_idx_sorted = sorted(significant_idx, key=lambda tup: tup[1])

for i in significant_idx_sorted:
    print(f'{df.columns.to_list()[i[0]]} p-value: {i[1]}')
    
# To increase f1 of my models, I will only use the k(17) best features from here out
selector = SelectKBest(f_classif, k = 17)
selector.fit_transform(features, labels)
scores = sorted(zip(all_features[1:], selector.scores_), key = lambda x: x[1], reverse=True)
best_features = list(map(lambda x: x[0], scores))[0:14]
best_features_list = poi_label + best_features

#use the best features as chosen by SelectKbest
data = featureFormat(my_dataset, best_features_list)

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, random_state=42)
    
sss = StratifiedShuffleSplit(n_splits= 100, test_size= 0.3, random_state= 42)
sss.get_n_splits(features, labels)

### here's where we actually try the variety of classifiers

#gaussian NB
g_clf = GaussianNB()
g_clf.fit(features_train, labels_train)
g_pred = g_clf.predict(features_test)

print("############################################")
print("Precision score_Gaussian is:", precision_score(labels_test, g_pred, average = 'macro', zero_division = 0))
print("Recall score_Gaussian is: ", recall_score(labels_test, g_pred, average = 'macro', zero_division = 0))

#Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(features_train, labels_train)
dt_pred = dt_clf.predict(features_test)

print("############################################")
print("Precision score_DecisionTree is:", precision_score(labels_test, dt_pred, average = 'macro', zero_division = 0))
print("Recall score_DecisionTree is: ", recall_score(labels_test, dt_pred, average = 'macro', zero_division = 0))

#Kneighbors
knn_clf = KNeighborsClassifier()
knn_clf.fit(features_train, labels_train)
knn_pred = knn_clf.predict(features_test)

print("############################################")
print("Precision score_KNN is:", precision_score(labels_test, knn_pred, average = 'macro', zero_division = 0))
print("Recall score_KNN is: ", recall_score(labels_test, knn_pred, average = 'macro', zero_division = 0))

#SVC
svc_clf = SVC(probability=False)
svc_clf.fit(features_train, labels_train)
svc_pred = svc_clf.predict(features_test)

print("############################################")
print("Precision score_SVC is:", precision_score(labels_test, svc_pred, average = 'macro', zero_division = 0))
print("Recall score_SVC is: ", recall_score(labels_test, svc_pred, average = 'macro', zero_division = 0))

#adaboost
ada_clf = AdaBoostClassifier(random_state=42)
ada_clf.fit(features_train, labels_train)
ada_pred = ada_clf.predict(features_test)

print("############################################")
print("Precision score_ADA is:", precision_score(labels_test, ada_pred, average = 'macro', zero_division = 0))
print("Recall score_ADA is: ", recall_score(labels_test, ada_pred, average = 'macro', zero_division = 0))

ada_clf.get_params().keys()

##Attempting tuning with scaling. using pipelines, applying MinMaxScaler() and PCA.

sss = StratifiedShuffleSplit(n_splits= 100, test_size= 0.3, random_state= 42)
scoring = 'f1'
skb = SelectKBest(f_classif)

scaler = MinMaxScaler()
pca = PCA()

g_clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

g_pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca),('skb', skb), ('gaussian', g_clf)])

g_params = dict(pca__n_components=[4,5,6,7,8,9,10],
                    skb__k = [1,2,3,4])

g_gs = GridSearchCV(g_clf, {}, cv = sss, scoring = scoring)

# Output

g_gs.fit(features, labels)
print("####################")
print("Best estimator:")
print(g_gs.best_estimator_)
print("Best score:")
print(g_gs.best_score_)
print("Best parameters:")
print(g_gs.best_params_)

g_clf = g_gs.best_estimator_
g_clf.fit(features_train, labels_train)

g_pred = g_clf.predict(features_test)

print("Precision score:")
print(precision_score(labels_test, g_pred, average = 'macro', zero_division = 0))
print("Recall score:")
print(recall_score(labels_test, g_pred, average = 'macro', zero_division = 0))
print("#####################")
print(' ')

gt_gs = GridSearchCV(g_pipe, g_params, cv = sss, scoring = scoring)

# Output

gt_gs.fit(features, labels)
print("####################")
print("Best estimator:")
print(gt_gs.best_estimator_)
print("Best score:")
print(gt_gs.best_score_)
print("Best parameters:")
print(gt_gs.best_params_)

gt_clf = gt_gs.best_estimator_
gt_clf.fit(features_train, labels_train)

gt_pred = gt_clf.predict(features_test)

print("Precision score:")
print(precision_score(labels_test, gt_pred, average = 'macro', zero_division = 0))
print("Recall score:")
print(recall_score(labels_test, gt_pred, average = 'macro', zero_division = 0))
print("#####################")
print(' ')

dt_clf = DecisionTreeClassifier(random_state=42)
#('pca', pca)
dt_pipe = Pipeline([('DecisionTree', dt_clf)])

dt_params = {
 'dt__splitter': ['best', 'random'],
 'dt__min_samples_split': [2, 3, 4, 5, 6, 7],
 'dt__max_features': labels
 }

dt_gs = GridSearchCV(dt_clf, {}, cv = sss, scoring = scoring)

# Output

dt_gs.fit(features, labels)
print("####################")
print("Best estimator:")
print(dt_gs.best_estimator_)
print("Best score:")
print(dt_gs.best_score_)
print("Best parameters:")
print(dt_gs.best_params_)

dt_clf = dt_gs.best_estimator_
dt_clf.fit(features_train, labels_train)

dt_pred = dt_clf.predict(features_test)

print("Precision score:")
print(precision_score(labels_test, dt_pred, average = 'macro', zero_division = 0))
print("Recall score:")
print(recall_score(labels_test, dt_pred, average = 'macro', zero_division = 0))
print("#####################")
print(' ')

ada_clf = AdaBoostClassifier()

ada_pipe = Pipeline([('AdaBoost', ada_clf)])

ada_params = {'n_estimators': [100], 
'algorithm': ['SAMME', 'SAMME.R'], 
'learning_rate': [.2, .5, 1, 1.4, 2], 
'random_state': [42]
}

ada_gs = GridSearchCV(ada_clf, {}, cv = sss, scoring = scoring)

# Output

ada_gs.fit(features, labels)
print("####################")
print("Best estimator:")
print(ada_gs.best_estimator_)
print("Best score:")
print(ada_gs.best_score_)
print("Best parameters:")
print(ada_gs.best_params_)

ada_clf = ada_gs.best_estimator_
ada_clf.fit(features_train, labels_train)

ada_pred = ada_clf.predict(features_test)

print("Precision score:")
print(precision_score(labels_test, ada_pred, average = 'macro', zero_division = 0))
print("Recall score:")
print(recall_score(labels_test, ada_pred, average = 'macro', zero_division = 0))
print("#####################")
print(' ')

adat_gs = GridSearchCV(ada_pipe, ada_params, cv = sss, scoring = scoring, refit=True)

# Output

adat_gs.fit(features, labels)
print("####################")
print("Best estimator:")
print(adat_gs.best_estimator_)
print("Best score:")
print(adat_gs.best_score_)
print("Best parameters:")
print(adat_gs.best_params_)

adat_clf = adat_gs.best_estimator_
adat_clf.fit(features_train, labels_train)

adat_pred = adat_clf.predict(features_test)

print("Precision score:")
print(precision_score(labels_test, adat_pred, average = 'macro', zero_division = 0))
print("Recall score:")
print(recall_score(labels_test, adat_pred, average = 'macro', zero_division = 0))
print("#####################")
print(' ')


test_classifier(ada_clf, my_dataset, best_features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

### Save the best performing classifier as clf for export
clf = nb_clf

### Save best features as features_list for export
features_list = select_k_best(4)

### Save to my_dataset for export
my_dataset = data_dict

dump_classifier_and_data(clf, my_dataset, features_list)



