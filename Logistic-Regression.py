import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("heart_2020_cleaned (1).csv")
X = dataset.drop("HeartDisease",axis = 1)
y = dataset["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lregmodel = LogisticRegression(solver='lbfgs', max_iter=300000)
lregmodel.fit(X_train,y_train)
predictions = lregmodel.predict(X_test)
classification_report(y_test,predictions)
classification_report(y_test,predictions)
confusion_matrix(y_test,predictions)