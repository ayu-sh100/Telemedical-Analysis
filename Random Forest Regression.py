import dataset as dataset
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from jedi.api.refactoring import inline
%matplotlib inline
dataset = pd.read_csv("heart_2020_cleaned_(1).csv"
X = dataset.drop("HeartDisease",axis = 1)
y = dataset["HeartDisease"]
from sklearn.model_selection import train_test_split
def train_test_split(X, y, test_size=0.3, random_state=1):
    pass
train_test_split(X, y, test_size=0.3, random_state=1)
class Y_train:
    pass
class Y_test:
    pass
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42, n_jobs=-1,
model.fit(X_train, y_train)
def RandomForestRegressor(random_state, n_jobs=-1):
    pass
RandomForestRegressor(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
from sklearn.model_selection import GridSearchCV
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
