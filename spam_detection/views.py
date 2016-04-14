# code related imports
# import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

from django.conf import settings
from django.shortcuts import render
from django.shortcuts import render_to_response
from django.template import RequestContext
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier



def index(request):
	file_path = settings.STATIC_ROOT + '/spambase.data'
	context = get_spam_data(file_path)
	return render_to_response('index.html', context, context_instance=RequestContext(request))


def rec_pre(predict_y, test_y):

    true_negative = 0.0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    
    for i, val in enumerate(predict_y):
        if val == 1: # positive
            if test_y[i] == 1: # true
                true_positive += 1
            else:
                false_positive += 1
        else: #negative
            if test_y[i] == 0: #true
                true_negative += 1
            else:
                false_negative += 1
    recall = true_positive/ (true_positive + false_negative)
    precision = true_positive/ (true_positive + false_positive)
    return recall, precision


def get_spam_data(file_path):
    k = open(file_path).read()
    result = {}
    percentage = 80.0
    k1 = k.split('\n')[:-1]
    random.shuffle(k1)
    length = len(k1)
    index = int((length * percentage) / 100)
    training_data = k1[:index]  # 80 - train
    test_data = k1[index:]      # 20 - test
    for i, val in enumerate(training_data):
        training_data[i] = map(float, val.split(','))

    for i, val in enumerate(test_data):
        test_data[i] = map(float, val.split(',')) # string to float value conversion

    training_x = []
    training_y = []
    test_x = []
    test_y = []
    for val in training_data:
        training_x.append(val[:-1])
        training_y.append(val[-1])

    for val in test_data:
        test_x.append(val[:-1])
        test_y.append(val[-1])

    # -----   Multinomial Naive Baiyes  -----
    clf1 = MultinomialNB()
    clf1.fit(training_x, training_y)
    a1 = clf1.score(test_x, test_y)
    predict_y = clf1.predict(test_x)
    recall1, precision1 = rec_pre(predict_y, test_y)
    result["multinomial_nb"] = {"accuracy": a1, "recall": recall1, 
    	"precision": precision1}

    # -----   Support Vector Machines  -----
    clf2 = svm.SVC()
    clf2.fit(training_x, training_y)
    a2 = clf2.score(test_x, test_y)
    predict_y = clf2.predict(test_x)
    recall2, precision2 = rec_pre(predict_y, test_y)
    result["svm"] = {"accuracy": a2, "recall": recall2, 
    	"precision": precision2}

    # -----   Decision Tree  -----
    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(training_x, training_y)
    a3 = clf3.score(test_x, test_y)
    predict_y = clf3.predict(test_x)
    recall3, precision3 = rec_pre(predict_y, test_y)
    result["decision_tree"] = {"accuracy": a3, "recall": recall3, 
    	"precision": precision3}

    # -----   GaussianNB  -----
    clf4 = GaussianNB()
    clf4.fit(training_x, training_y)
    a4 = clf4.score(test_x, test_y)
    predict_y = clf4.predict(test_x)
    recall4, precision4 = rec_pre(predict_y, test_y)
    result["gaussian_nb"] = {"accuracy": a4, "recall": recall4, 
    	"precision": precision4}

    # -----   LogisticRegression  -----
    clf5 = LogisticRegression()
    clf5.fit(training_x, training_y)
    a5 = clf5.score(test_x, test_y)
    predict_y = clf5.predict(test_x)
    recall5, precision5 = rec_pre(predict_y, test_y)
    result["logistic_regression"] = {"accuracy": a5, "recall": recall5, 
    	"precision": precision5}

    # -----   RandomForest  -----
    clf6 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
    clf6.fit(training_x, training_y)
    a6 = clf6.score(test_x, test_y)
    predict_y = clf6.predict(test_x)
    recall6, precision6 = rec_pre(predict_y, test_y)
    result["random_forest"] = {"accuracy": a6, "recall": recall6, 
    	"precision": precision6}

    # -----  GradientBoosting -----
    clf7 = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=10, random_state=0)
    clf7.fit(training_x, training_y)
    a7 = clf7.score(test_x, test_y)
    predict_y = clf7.predict(test_x)
    recall7, precision7 = rec_pre(predict_y, test_y)
    result["gradient_boosting"] = {"accuracy": a7, "recall": recall7, 
    	"precision": precision7}

    #  -----   AdaBoost  -----
    clf8 = AdaBoostClassifier(n_estimators=200)
    clf8.fit(training_x, training_y)
    a8 = clf8.score(test_x, test_y)
    predict_y = clf8.predict(test_x)
    recall8, precision8 = rec_pre(predict_y, test_y)
    result["adaboost"] = {"accuracy": a8, "recall": recall8, 
    	"precision": precision8}

    return result
    # Graph plotting to not do on the fly. Not a good practise.
    # Can use frontend libraries for plotting. 
    # Can do it if we do get the time
    # print " ---------------- Plot the Graph ------------------ "
    # plt.title('Accuracy, Recall, Precision')
    # plt.plot([a1,a2,a3,a4,a5,a6,a7,a8],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall1,recall2,recall3,recall4,recall5,recall6,recall7,recall8], 'g--',marker='o', label='Recall', linewidth=1)
    # plt.plot([precision1,precision2,precision3,precision4,precision5,precision6,precision7,precision8], 'b--',marker='o', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # print " ---------------- Plot all 8 Graphs ------------------ "
    # plt.title('Multinomial Naive Baiyes')
    # plt.plot([a1],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall1], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision1], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('Support Vector Machines')
    # plt.plot([a2],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall2], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision2], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('Decision Tree')
    # plt.plot([a3],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall3], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision3], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('GaussianNB')
    # plt.plot([a4],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall4], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision4], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('LogisticRegression')
    # plt.plot([a5],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall5], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision5], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('RandomForest')
    # plt.plot([a6],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall6], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision6], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('GradientBoosting')
    # plt.plot([a7],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall7], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision7], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()

    # plt.title('AdaBoost')
    # plt.plot([a8],color='red',marker='o',label = 'Accuracy', linewidth=2)
    # plt.plot([recall8], 'g--',marker='^', label='Recall', linewidth=1)
    # plt.plot([precision8], 'b--',marker='*', label='Precision', linewidth=1)
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()