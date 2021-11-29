from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

# Leggo il file CSV
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding = 'utf-8'
)

# Salvo le features del dataset
FEATURES = dataFrame.columns.values
FEATURES = np.append(FEATURES[0:2], FEATURES[5:-2])
N_FEATURES = int(len(FEATURES))

# Inizializzo la matrice X
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES + 3))].values

# Inizializzo il vettore target
y = dataFrame.iloc[:, N_FEATURES + 1].values
# 0 traffico normale, 1 traffico malevole

# Standardizzo i valori delle features
X = StandardScaler().fit_transform(X)

# Divido il dataset in 70% dataset di traning e 30% dataset di test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

start_time = time.time()

# Creiamo un'istanza di RandomForest composta da 100 alberi di decisione
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100)

# Addestriamo l'algoritmo sul dataset di training
clf.fit(X_train, y_train)

# Eseguiamo l'algoritmo sul dataset di testing
y_pred = clf.predict(X_test)

# Confronto il vettore target con il vettore delle predizioni
from sklearn import metrics
print("\nAccuracy by considering all features: " + str(metrics.accuracy_score(y_test, y_pred)))

# Stampo il tempo di esecuzione dell'algoritmo
print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

# Visualizzo le prime 10 features più importanti del dataset
from util.util import plotFeatureImportances
plotFeatureImportances(clf, FEATURES, 'image/feature-importances.png', 10)

# Estraggo un dataframe più piccolo utilizzando solo le features più importanti
from util.util import minimizeDataFrameByFeatureImportances
minDataFrame = minimizeDataFrameByFeatureImportances(clf, FEATURES, 
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv', .9)

# Ripeto l'addestramento utilizzando il dataframe aggiornato
FEATURES = minDataFrame.columns.values
N_FEATURES = int(len(FEATURES))

X = minDataFrame.iloc[:, 0:N_FEATURES].values
# Il vettore y non ha bisogno di subire nessuna modifica

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

start_time = time.time()

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nAccuracy by considering the best " + str(N_FEATURES)
    + " features: " + str(metrics.accuracy_score(y_test, y_pred)))
print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

# Visualizzo di nuovo le prime 10 features più importanti per vedere di quanto variano
# i valori dell'impurità per ogni features
plotFeatureImportances(clf, FEATURES, 'image/feature-importances-after-reducing.png', 10)

"""
# Addestriamo l'algoritmo sul dataset modificato dalla PCA
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set_PCA.csv',
    encoding = 'utf-8'
)

FEATURES = dataFrame.columns.values
N_FEATURES = int(len(FEATURES)) - 1
FEATURES = FEATURES[1 : N_FEATURES]

X = dataFrame.iloc[:, np.arange(1, N_FEATURES)].values
y = dataFrame.iloc[:, N_FEATURES].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

start_time = time.time()

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy after PCA: " + str(metrics.accuracy_score(y_test, y_pred)))

print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')
"""