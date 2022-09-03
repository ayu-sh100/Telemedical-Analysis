import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
dataset = pd.read_csv("heart_2020_cleaned (1).csv")
dataset.head()
len(dataset)
X = dataset.drop("HeartDisease",axis = 1)
y = dataset["HeartDisease"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
lregmodel = LogisticRegression(solver='lbfgs', max_iter=300000)
lregmodel.fit(X_train,y_train)
predictions = lregmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)