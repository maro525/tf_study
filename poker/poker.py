from math import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import re,os
data=pd.read_csv('poker_train.csv')
x=data[['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5']]
y=data['c']
clf=RandomForestClassifier(n_estimators=200, min_samples_split=1)
clf.fit(x,y)
print clf.score(x,y)
test=pd.read_csv('poker_test.csv')
x=test[['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5']]
y=test['c']
p=clf.predict(x)
print clf.score(x,y)
'''
print clf.feature_importances_
t=np.arange(0.0,100.0)
plt.plot(t,test['quality'],'--',t,p,'-')
plt.show()
'''
