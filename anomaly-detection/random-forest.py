from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

start_time = time.time()

# Leggo il file CSV
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding = 'utf-8'
)

# Salvo le features del dataset
FEATURES = dataFrame.columns.values
FEATURES = np.append(FEATURES[0:2], FEATURES[5:int(len(FEATURES) - 2)])
N_FEATURES = int(len(FEATURES))

# Inizializzo la matrice X
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES + 3))].values

# Inizializzo il vettore target
y = dataFrame.iloc[:, N_FEATURES + 1].values
# 0 traffico normale, 1 traffico malevole

# Standardizzo i valori delle features
X = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

"""
from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
"""

feature_imp = pd.Series(clf.feature_importances_, 
    index = FEATURES).sort_values(ascending = False)

print(feature_imp)

print('\nTime elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')