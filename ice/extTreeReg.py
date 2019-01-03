from math import *
import pandas as pd
import numpy as np
from sklearn.tree import ExtraTreeRegressor
import matplotlib.pyplot as plt
import re,os
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=ExtraTreeRegressor()
clf.fit(x,y)
p=clf.predict(x)
print clf.score(x,y)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--',t,p,'-')
plt.show()
