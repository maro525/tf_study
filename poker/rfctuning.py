from math import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import *
import matplotlib.pyplot as plt
import re,os
data=pd.read_csv('poker_train.csv')
x=data[['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5']]
y=data['c']
clf =RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
	'n_estimators': [700,750,800],
	'min_samples_split' : [2, 3, 5],
        'max_depth'         : [3, 5, 10, 15, 17],
	'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
CV_rfc.fit(x, y)
print CV_rfc.best_params_
test=pd.read_csv('poker_test.csv')
x=test[['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5']]
y=test['c']
p=CV_rfc.predict(x)
print CV_rfc.score(x,y)
