# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:21:49 2020

@author: patno_000
"""
import pandas as pd
import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
bc = ds.load_breast_cancer()

type(bc)

dir(bc)

bc.DESCR
bc.data
bc.feature_names
bc.filename
bc.target
bc.target_names

X = pd.DataFrame(bc.data, columns = bc.feature_names)
y = pd.DataFrame(bc.target, columns = ['value'])
X.head()
y.head()

X_ = X
y_ = y.values.ravel()

# In[]

from sklearn.model_selection import train_test_split
SEED = 1220
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.25, random_state=SEED)


X_train.columns

# In[]
for i, col in enumerate(X.columns):
    
    plt.scatter(X[col], X['mean radius'], c=y.values.ravel())
    plt.xlabel(col)
    plt.ylabel('mean radius')
#    plt.legend()
    plt.show()

# In[]
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV as CV

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as CR

# In[]
"""
 logreg
"""
LR = LogisticRegression(random_state = SEED, solver='liblinear', max_iter=500)
LR.get_params()

LR.fit(X_train, y_train)

y_pred_LR = LR.predict(X_test)

# In[]

print(accuracy_score(y_test, y_pred_LR))
LR_acc = accuracy_score(y_test, y_pred_LR)
print(CR(y_test, y_pred_LR, output_dict=True))
# very good predictor in this case
LR_CR = CR(y_test, y_pred_LR, output_dict=True)
LR_F1 = LR_CR['weighted avg']['f1-score']
#logreg does really well here

# In[]

"""
Decision Tree Classifier
"""
DT = DecisionTreeClassifier()

DT_hyp = {
            'max_depth':[1,2,3,4,5,6,7],
            'min_samples_leaf':[0.01,0.05,0.1,0.15,0.2,0.5,2],
            'max_features':['sqrt','log2',None,0.2,0.5,0.7]            
        }

DT_CV = CV(estimator = DT,
           param_grid = DT_hyp,
           scoring = 'accuracy',
           cv = 5)

DT_CV.fit(X_train, y_train)

# In[]
DT_best = DT_CV.best_estimator_
print(DT_best.get_params())

y_pred_DT = DT_best.predict(X_test)

# In[]

DT_acc = accuracy_score(y_test, y_pred_DT)
DT_CR = CR(y_test, y_pred_DT, output_dict = True)
print(DT_acc)
print(DT_CR)
print(DT_CR['weighted avg']['f1-score'])

# a little worse than logreg

# In[]
 
"""
    KNN
"""
KNN = KNeighborsClassifier()

KNN_hyp = {
        'n_neighbors':[*np.arange(2,25,1)]
        }

KNN_CV = CV(estimator = KNN,
            param_grid = KNN_hyp,
            scoring = 'accuracy',
            cv = 5)

KNN_CV.fit(X_train, y_train)

# In[]
KNN_best = KNN_CV.best_estimator_
print(KNN_best.get_params())

y_pred_KNN = KNN_best.predict(X_test)

# In[]

KNN_acc = accuracy_score(y_test, y_pred_KNN)
KNN_CR = CR(y_test, y_pred_KNN, output_dict=True)
print(KNN_acc)
print(KNN_CR)
print(KNN_CR['weighted avg']['f1-score'])

# worst so far

# In[]

"""
  Naive Bayes
"""
NB = GaussianNB()
NB.fit(X_train, y_train)

y_pred_NB = NB.predict(X_test)

# In[]

NB_acc = accuracy_score(y_test, y_pred_NB)
NB_CR = CR(y_test, y_pred_NB, output_dict=True)
print(NB_acc)
print(NB_CR)
print(NB_CR['weighted avg']['f1-score'])

# still not great
# In[]

classifiers = [('Logistic Regression', LR_acc), ('Classification Tree', DT_acc),
               ('K Nearest Neighbours', KNN_acc), ('Naive Bayes', NB_acc)]


for clf_name, clf_acc in classifiers:    
   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, clf_acc))


# In[]
"""
ensemble
"""
from sklearn.ensemble import VotingClassifier as VC

classifiers = [('Logistic Regression', LR), ('Classification Tree', DT),
               ('K Nearest Neighbours', KNN), ('Naive Bayes', NB)]


vc1 = VC(estimators=classifiers)     

# Fit vc to the training set
vc1.fit(X_train, y_train)   

# Evaluate the test set predictions
y_pred_vc1 = vc1.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred_vc1)
print('Voting Classifier: {:.3f}'.format(accuracy))

# actually does worse thanonly LogReg, but better than DT,KNN,NB on their own
# In[]
"""
ensemble without logreg
"""
classifiers = [('Classification Tree', DT),
               ('K Nearest Neighbours', KNN), ('Naive Bayes', NB)]

vc2 = VC(estimators=classifiers)     

# Fit vc to the training set
vc2.fit(X_train, y_train)   

# Evaluate the test set predictions
y_pred_vc2 = vc2.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred_vc2)
print('Voting Classifier: {:.3f}'.format(accuracy))

#does better than DT, KNN, NB on their own, LR on its own still superior in this example
