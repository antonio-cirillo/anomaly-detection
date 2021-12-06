from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

# Leggo il file CSV
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW-NB15_SAMPLING.csv',
    encoding = 'utf-8'
)

# Salvo in un'array le labels
LABELS = list(dict.fromkeys(dataFrame['attack_cat'].values))
LABELS[-1] = 'Normal'

# Elimino tutti parametri non numerici
from util.util import deleteStringColumn
dataFrame = deleteStringColumn(dataFrame)

# Salvo le features del dataset
FEATURES = dataFrame.columns.values
N_FEATURES = int(len(FEATURES))

# Inizializzo la matrice X
X = dataFrame.iloc[:, np.arange(0, N_FEATURES - 1)].values

# Inizializzo il vettore target
y = dataFrame.iloc[:, N_FEATURES - 1].values

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
plotFeatureImportances(clf, FEATURES[0:-1], 'image/feature-importances.png', 10)

# Memorizzo la lista delle features in ordine di importanza (riduzione massima di impurità)
feature_imp = clf.feature_importances_

# Estraggo un dataframe più piccolo utilizzando solo le features più importanti
from util.util import minimizeDataFrameByFeatureImportances
for i in [.3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]:
    FEATURES = dataFrame.columns.values
    minDataFrame = minimizeDataFrameByFeatureImportances(feature_imp, FEATURES[0:-1], 
        os.getcwd() + '\\dataset\\UNSW-NB15_SAMPLING.csv', i)
    
    # Ripeto l'addestramento utilizzando il dataframe aggiornato
    FEATURES = minDataFrame.columns.values
    N_FEATURES = int(len(FEATURES))

    X = minDataFrame.iloc[:, 0:N_FEATURES].values    
    X = StandardScaler().fit_transform(X)    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    start_time = time.time()

    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\nAccuracy by considering the best " + str(N_FEATURES)
        + " features: " + str(metrics.accuracy_score(y_test, y_pred)))
    print('Time elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

    COUNT_ERROR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TYPE_ERROR = [[], [], [], [], [], [], [], [], [], []]

    for row_index, (input, prediction, label) in enumerate(zip(X_test, y_pred, y_test)):
        if prediction != label:
            # print('Row', row_index, 'has been classified as', prediction, 'and should be', label)
            COUNT_ERROR[int(label)] += 1
            TYPE_ERROR[int(label)].append(prediction)

    for index in np.arange(int(len(LABELS))):
        TYPE_ERROR[index] = list(dict.fromkeys(TYPE_ERROR[index]))

    for index in np.arange(int(len(LABELS))):
        print('Label', LABELS[index], 'has encountered a number of prediction error equals to', COUNT_ERROR[index])
        if COUNT_ERROR[index] > 0:
            print(LABELS[index], 'was incorrectly predicted as:')
            for error in TYPE_ERROR[index]:
                print(LABELS[error])

    break