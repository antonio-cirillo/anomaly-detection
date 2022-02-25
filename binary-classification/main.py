from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from util.util import minimize_by_feature_importance
from util.util import plot_feature_importance

import pandas as pd
import numpy as np
import time
import os

# Read file csv
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding='utf-8'
)

# Save dataset features
FEATURES = dataFrame.columns.values
FEATURES = np.append(FEATURES[0:2], FEATURES[5:-2])
N_FEATURES = int(len(FEATURES))

# Create matrix X
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES + 3))].values

# Create vector y
y = dataFrame.iloc[:, N_FEATURES + 1].values

# Standardize value of matrix X
X = StandardScaler().fit_transform(X)

# Split X and y for testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Start timing
start_time = time.time()

# Create and fit the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Test random forest
y_pred = clf.predict(X_test)

# Measure accuracy
print("\nAccuracy by considering all features: " + str(metrics.accuracy_score(y_test, y_pred)))

# Print time
print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

# View ten first features that minimize impurity
plot_feature_importance(clf, FEATURES, 'image/feature-importance.png', 10)

# Extract sub dataframe with only first ten feature
minDataFrame = minimize_by_feature_importance(clf, FEATURES,
                                              os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv', .9)

# Repeat same operation
FEATURES = minDataFrame.columns.values
N_FEATURES = int(len(FEATURES))

X = minDataFrame.iloc[:, 0:N_FEATURES].values
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

start_time = time.time()

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nAccuracy by considering the best " + str(N_FEATURES) +
      " features: " + str(metrics.accuracy_score(y_test, y_pred)))
print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')
