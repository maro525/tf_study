import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=RandomForestClassifier(n_estimators=200, min_samples_split=1)
clf.fit(x,y)
print clf.score(x,y)
print clf.feature_importances_
