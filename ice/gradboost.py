import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf = GradientBoostingRegressor(n_estimators=1000, learning_rate=1.2,max_depth=1, random_state=0)
clf.fit(x,y)
print clf.score(x,y)
p=clf.predict(x)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],':b')
plt.plot(t,p,'-b')
plt.legend(('real','gradboost'))
plt.show()
