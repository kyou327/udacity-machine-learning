# Enron Fraud Detection
 
The project uses the enron dataset, which includes the email data and financial data of employees. This project uses machine learning algorithms to identify Enron Employees who may have committed fraud. 

## Scope of this submission
This submission contains the following files:

| File / Directory | Description |
| ---------------- | ----------- |
| poi_id.py        | Python code used for the analysis. It is packaged with the other required files.|
| my_classifier.pkl, my_dataset.pkl, my_feature_list.pkl | required pickle files |
| project.md | the project documention |

## Question 1:
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
 
The dataset includes Enron employees’ email data and financial data. The project applies machine learning algorithm to discover patterns in the data in order to detect fraud. The dataset contains in total 146 data points, namely 146 employees data. The features in the dataset can be grouped into three categories: Financial features, Email features and POI label. The POI label is referring to "person of interest" the main datapoint to categorize the people guilty of fraud. The dataset has these persons listed as a boolean category, so we already know who is under scrutiny, and can see how our accuracy measures up in the code.

First of all, the project takes a closer look at the dataset: total data points, allocation across classes, features used, and missing values. In the following a summary of the data exploration:

| Data                         | Value |
|:---------------------------- | -----:|
| Total number of data points: | 146 |
| Number of person of interest: | 18 |
| Number of non person of interest: | 128 |
| Number of features used | 24 |
| Number of missing value in feature salary: | 51 |
| Number of missing value in feature to_messages: | 60 |
| Number of missing value in feature deferral_payments: | 107 |
| Number of missing value in feature total_payments: | 21 |
| Number of missing value in feature exercised_stock_options: | 44 |
| Number of missing value in feature bonus: | 64 |
| Number of missing value in feature restricted_stock: | 36 |
| Number of missing value in feature shared_receipt_with_poi: | 60 |
| Number of missing value in feature restricted_stock_deferred: | 128 |
| Number of missing value in feature total_stock_value: | 20 |
| Number of missing value in feature expenses: | 51 |
| Number of missing value in feature loan_advances: | 142 |
| Number of missing value in feature from_messages: | 60 |
| Number of missing value in feature from_this_person_to_poi: | 60 |
| Number of missing value in feature director_fees: | 129 |
| Number of missing value in feature deferred_income: | 97 |
| Number of missing value in feature long_term_incentive: | 80 |
| Number of missing value in feature from_poi_to_this_person: | 60 |

After the ‘TOTAL’ is removed, we plot the salary and total_payments again and can see from the plot that there are three outliers in terms of salary.

Those three persons happen to be the three big bosses of Enron at that time...

>[('SKILLING JEFFREY K', 1111258),
> ('LAY KENNETH L', 1072321),
> ('FREVERT MARK A', 1060932)]


Two of the three persons (SKILLING JEFFREY K & LAY KENNETH L) are persons of interest out of 18 persons of interest, so we will not remove these three outliers.


## Question 2:
> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

The project creates 5 new features: **'bonus_total_ratio', 'bonus_ratio', 'emails_to_crooks_ratio',** and **'emails_from_crooks_ratio','poi_message_percent'**. Instead of comparing with absolute numbers, the ratio shows better how strong the email connection of this person to the person of interest than to the non person of interest. After the new features are created, the project applies **SelectKBest** to determine the strength of all features. In the following we see that the features **'from_this_person_to_poi'** and **'bonus'** are quite strong features. **'bonus_ratio'** was also a relatively strong feature for selection.

| Feature | Score |
|:------- | -----:|
|'other' | 957.2535461876616 |
|'from_this_person_to_poi' | 369.7532172430564 |
|'bonus' | 106.48203651244525 |
|'deferred_income' | 98.80556031148178 |
|'bonus_ratio' | 73.80409428481394 |
|'emails_to_crooks_ratio' | 51.334851098133 |
|'bonus_total_ratio' | 42.1656034215212 |
|'expenses' | 33.486193011637155 |
|'from_messages' | 27.106135665051944 |
|'emails_from_crooks_ratio' | 23.22496739193553 |
|'from_poi_to_this_person' | 21.362274071678936 |
|'to_messages' | 12.727758976027461 |
|'shared_receipt_with_poi' | 8.616173410395897 |
|'total_stock_value' | 8.293372261543798 |
|'exercised_stock_options' | 8.029519666394583 |
|'long_term_incentive' | 7.86466351381628 |
|'total_payments' | 5.071719761621356 |
|'poi' | 2.132975551580203 |
|'restricted_stock' | 1.7550223987098177 |
|'poi_message_percent' | 1.488246748788039 |
|'deferral_payments' | 0.6839240758399686 |
|'restricted_stock_deferred' | 0.016057584422086237 |
|'director_fees' | 0.013739813158417808 |

the project deploys univariate feature selection to select the k best features:

1. Create function with **SelectKBest** to select the k best features
1. Use **GridSearchCV** to select the best parameters
1. Set the parameters of the selected machine learning algorithms with the best parameters and evaluate the performance of the algorithms
1. Repeat the above process for k ranged from 3 to 22. Based on the Recall, select the best K value.



| Feature | Score |
|:------- | -----:|
|'other' | 957.2535461876616 |
|'from_this_person_to_poi' | 369.7532172430564 |
|'bonus' | 106.48203651244525 | 
|'deferred_income' | 98.80556031148178 |

Some of the classifiers were tuned with MinMaxScaler in order to moderate the ranges between different features. Some, like 'salary' have very high number ranges in the millions, and some like 'from_this_person_to_poi' have less than 1000.


## Question 3:
> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [relevant rubric item: “pick an algorithm”]

**Naive bayes**, **support vector classifier**, **K nearest neighbors**, **decision tree** and **adaboost** were tested with the selected k_best_features. Select K Best settled on 17 ideal features to evaluate. For evaluation strategy the accuracy_score, precision_score, recall_score and f1_score will be calculated.


To decide which algorithm works better, the project will mainly focus on precision and recall. The precision and recall which are greater than 0.3 will also be considered as good performance. The accuracy_score here is not a strong index for the performance, due to that there is only small percent of person of interest in the dataset. 

The evaluation strategy:

1. Different K value are tried with **SelectKBest**, within executable range.
1. Algorithms with precision_score and recall_score greater than 0.3 are selected.
1. The higher the recall is, the better the performance is.

|Top Algorithm|
|-------------|
|Adaboost Classifier (without Tuning) |
|Accuracy: 0.789 |
|Precision: 0.538 |
|Recall: 0.37 |
|F1 score: 0.438 |

|Decision Tree Classifier (without Tuning) |
|Accuracy: 0.696 |
|Precision: 0.269 |
|Recall: 0.213 |
|F1 score: 0.238 |

|Gaussian Naive Bayes Classifier (with Tuning) |
|Accuracy: 0.775|
|Precision: 0.488 |
|Recall: 0.198 |
|F1 score: 0.281 |





## Question 4:
> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

**Hyperparameter tuning** selects a set of optimal hyperparameters for machine learning algorithms. It can help to avoid overfitting and increase the performance of the algorithms on an independent dataset (Source: Wikipedia). 

- For **Gaussian Naive Bayes classifier**, **GridSearchCV** searches through different combination of applying scaling to **features** and selecting the **Select K Best** **n_components**.
-Other classifiers ran into errors due to the limited size of the dataset, so I witheld from tuning them further.
Below shows how they would have been tuned.

```
1) For **support vector classifier**, the parameter C, gamma and kernel are tuned as such:
```python
param_grid_svc = [
  {'C': [1, 10, 50, 100, 150, 1000], 'kernel': ['linear','rbf']},
  {'C': [1, 10, 50, 100, 150, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf']}]
```
2) For **decision tree classifier**, the parameter criterion, min_samples_split, max_depth, min_samples_leaf, and max_leaf_nodes are tuned as such:
```python
param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 10, 20],
    "max_depth": [None, 2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_leaf_nodes": [None, 5, 10, 20],
}
```

After parameter tuning, there were slight gains in **Naive Bayes classifier**, but these turned out to be overfitting issues which resulted in matching precision and recall scores.


## Question 5: 
> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? [relevant rubric items: “discuss validation”, “validation strategy”]

In machine learning, validation is a method to test the model’s performance by splitting the data into training and testing data. The model is trained with the training dataset and tested with the testing dataset. The performance on the testign dataset with validate the algorithm. With validation the model’s performance can be improved when it is applied with an independent dataset. Without proper validation of the machine learning algorithm, overfitting can occur: the model can be overfitted with every single data point. In this way the model could have very high accuracy_score on the training data. However, the model will fail with a new dataset, as the model doesn’t generalize the data points 'learned', but simply ‘remembers’ each single one.
 
This project applies the cross validation strategy: The dataset will be split into 10 folds, and each fold will be used both for testing as well as training. The performance of the model will be measured with the average accuracy_score, precision_score, recall_score and f1_score. The cross validation strategy should have more advantages than simply splitting the data into training and testing sets. The main reason is that the data points in the Enron dataset is in total 145 (after removing the ‘TOTAL’ line) and with cross validation the training and testing dataset can be better split across the category person of interest(POI).


## Question 6: 
> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
 
The project uses the following evaluation metrics: **accuracy_score**, **precision_score**, **recall_score**, and **f1_score**. When you take a closer look at the accuracy_score, most of them are above 0.75, some well above 0.80, which doesn’t tell much about the performance of the model. Moreover, since there are in total 18 persons of interest out of 145 in the dataset, the accuracy_score will not be a strong index for the performance of any one algorithm.

As the goal of the project is to identify the persons of interest, namely how many persons of interest are identified, the recall will be a better metric here: 

```python
recall_score = number_correct_identified_POI / total_number_POI
```

If the model predicts every person as POI, then the recall_score will be 1. Let's look at precision as well. The precision_score tells us how many predicted POI are true POI:

```python
precision_score = number_correct_identified_POI / total_number_POI_predicted
```
F1_score is the harmonic mean of precision and recall.

To sum up the evaluation strategy:

1. Use **SelectKBest** to identify the best feature use.
1. Algorithms with a precision_score and recall_score (and to a lesser degree, f1_score) greater than 0.3 are selected.
1. The higher the recall is, the better the performance of the algorithm is.

With the 3 best features selected, **Adaboost Classifier** showed the best performance upon testing with tester.py:

1. Accuracy = 0.789: 78.9% of the 146 data points are correctly predicted
1. Precision: 0.538: Among the identified Persons of Interest, 53.8% are true POI.
1. Recall: 0.37: Among the 18 Persons of Interest, 37% are correctly identified.
1. f1 score: 0.438: The harmonic mean of precision and recall is 0.438.


## Sources
- Udacity Data Analyst Nanodegree - Intro to Machine Learning
- https://github.com/jeremy-shannon/udacity-Intro-to-Machine-Learning/tree/master/k_means
- https://github.com/falodunos/intro-to-machine-learning-udacity-final-project/blob/master/README.md
- http://machinelearningmastery.com/feature-selection-machine-learning-python/
- http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
- http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
- http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
- https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)





