import dataset as dataset
import matplotlib
import numpy as np
import predictions as predictions
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
grid_search.fit(X_train, y_train)
grid_search.best_score_
rf_best = grid_search.best_estimator_rf_best
from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[5], feature_names = X.columns,class_names=['Disease', "No Disease"],filled=True);
from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[7], feature_names = X.columns,class_names=['Disease', "No Disease"],filled=True);
rf_best.feature_importances_
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})
imp_df.sort_values(by="Imp", ascending=False)
y_pred = model.predict(X_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
errors = abs(predictions - y_test)
#Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')