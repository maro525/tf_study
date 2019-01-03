from math import *
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import re,os
data=pd.read_csv('red.csv')
x=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y=data['quality']
clf=ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
clf.fit(x,y)
test=pd.read_csv('red_test.csv')
x=test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y=test['quality']
p=clf.predict(x)
print clf.score(x,y)
print clf.feature_importances_
t=np.arange(0.0,100.0)
plt.plot(t,test['quality'],'--',t,p,'-')
plt.show()
