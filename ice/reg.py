import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
x=sm.add_constant(x)
y=data['ice']
est=sm.OLS(y,x).fit()
print est.summary()
