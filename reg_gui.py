from math import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re,os
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
x=sm.add_constant(x)
y=data['ice']
est=sm.OLS(y,x).fit()
t=np.arange(0.0,31.0)
const,tem,st=est.params
plt.plot(t,data['ice'],':b')
plt.plot(t,est.predict(x),'-b')
plt.legend(('real','OLS'))
plt.show()
