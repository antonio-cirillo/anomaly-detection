from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Leggo il file CSV
dataFrame = pd.read_csv(
    os.getcwd() + '\\dataset\\UNSW_NB15_testing-set.csv',
    encoding = 'utf-8'
)

# Salvo le features del dataset
FEATURES = dataFrame.columns.values
N_FEATURES = int(len(FEATURES)) - 2

# Inizializzo la matrice X
X = dataFrame.iloc[:, np.append([0, 1], np.arange(5, N_FEATURES))].values

# Inizializzo il vettore target
targetDataFrame = dataFrame.iloc[:, N_FEATURES + 1]
# y = targetDataFrame.values
# y = np.where(y == 0, -1, 1)

# Standardizzo i valori delle features
X = StandardScaler().fit_transform(X)

# Valutiamo la PCA prendendo in considerazione le prime n componenti
N_COMPONENTS_TO_EVALUATE = 15
pca = PCA(n_components = N_COMPONENTS_TO_EVALUATE)
principalComponents = pca.fit_transform(X)

PCA_EXPLAINED_VARIANCE_RATIO = pca.explained_variance_ratio_.cumsum()
plt.plot(range(1, N_COMPONENTS_TO_EVALUATE + 1),
    PCA_EXPLAINED_VARIANCE_RATIO, marker = 'o')

plt.xlim(0, N_COMPONENTS_TO_EVALUATE + 1)
plt.ylim(0, 1)
plt.grid()

plt.xlabel('Numero delle componenti')
plt.ylabel('% Varianza')

plt.show()

"""
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(X)


principalDataFrame = pd.DataFrame(principalComponents,
    columns = ['principal component 1', 'principal component 2'])
finalDataFrame = pd.concat([principalDataFrame, targetDataFrame], axis = 1)

normalDataFrame = finalDataFrame[finalDataFrame['label'] == 0]
maliciousDataFrame = finalDataFrame[finalDataFrame['label'] == 1]

plt.scatter(normalDataFrame.iloc[:, 0], normalDataFrame.iloc[:, 1],
    color = 'blue', marker = 'x', label = 'normal traffic')

plt.scatter(maliciousDataFrame.iloc[:, 0], maliciousDataFrame.iloc[:, 1],
    color = 'red', marker = 'o', label = 'malicious traffic')

plt.xlabel('principal component 1')
plt.ylabel('principal compoentn 2')
plt.legend(loc = 'upper left')

plt.show()


print(pca.explained_variance_ratio_.cumsum())
"""