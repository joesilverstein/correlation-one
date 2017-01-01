# MachineLearningTest.py
# Joe Silverstein
# 10-6-16

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# os.chdir('/Users/joesilverstein/Google Drive/Correlation One')

df = read_csv('./data/stock_returns_base150.csv')

past = df[0:49]

train = past.sample(frac = 0.8)
test = past.drop(train.index)

X_train = train.ix[:, 2:10]
y_train = train.ix[:, 1]
X_test = test.ix[:, 2:10]
y_test = test.ix[:, 1]

est = GradientBoostingRegressor(loss = 'lad', criterion = 'mae').fit(X_train, y_train)

mae = mean_absolute_error(y_test, est.predict(X_test))

