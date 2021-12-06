import pandas as pd
import numpy as np
import os

ATTACK_CAT = ['Backdoors', 'Analysis', 'Fuzzers', 
    'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']

DATAFRAME_LIST = []

# Inizializzo i DataFrame
for i in np.arange(1, 2):
    dataFrame = pd.read_csv(
        os.getcwd() + '\\dataset\\UNSW-NB15_' + str(i) + '.csv',
        encoding = 'utf-8'
    )
    dataFrame['attack_cat'] = dataFrame['attack_cat'].str.strip()
    DATAFRAME_LIST.append(dataFrame.loc[dataFrame['Label'] == 0])
    attackDF_ = dataFrame.loc[dataFrame['Label'] == 1]
    attackDF_['attack_cat'].replace(to_replace = 'Backdoor', value = 'Backdoors')
    for value in ATTACK_CAT:
        DATAFRAME_LIST.append(attackDF_.loc[attackDF_['attack_cat'] == value])

# Aggiungo ai DataFrame per ogni tipologia di account i dati contenuti negli altri 3 file
for i in np.arange(2, 5):
    dataFrame = pd.read_csv(
        os.getcwd() + '\\dataset\\UNSW-NB15_' + str(i) + '.csv',
        encoding = 'utf-8'
    )
    index = 1
    dataFrame['attack_cat'] = dataFrame['attack_cat'].str.strip()
    DATAFRAME_LIST[0] = DATAFRAME_LIST[0].append(dataFrame.loc[dataFrame['Label'] == 0])
    attackDF_ = dataFrame.loc[dataFrame['Label'] == 1]
    for value in ATTACK_CAT:
        DATAFRAME_LIST[index] = DATAFRAME_LIST[index].append(
            attackDF_.loc[attackDF_['attack_cat'] == value])
        index += 1
            
# Unisco la lista Backdoor con la lista Backdoors
DATAFRAME_LIST[1] = DATAFRAME_LIST[1].append(DATAFRAME_LIST[2])
del DATAFRAME_LIST[2]
del ATTACK_CAT[0]

# Ordino i DATAFRAME dal più piccolo al più grande
DATAFRAME_LIST.sort(key = lambda d: d.shape[0])

# Salviamo il numero minimo di item all'interno di tutti i dataframe
minum = DATAFRAME_LIST[0].shape[0]

# Estraggo n righe dagli m dataframe per creare un nuovo dataframe
# n = minum
# m = |attack_cat|
sampleDataFrame = pd.DataFrame(columns = DATAFRAME_LIST[0].columns.values)

for i in range(int(len(DATAFRAME_LIST))):
    df = DATAFRAME_LIST[i]
    df['Label'] = df['Label'].map({1: (i + 1)}) 
    sampleDataFrame = sampleDataFrame.append(df.sample(n = minum * (i + 1)))

# Inserisco il valore 0 nelle celle non inizializzate
cols = sampleDataFrame.columns
# sampleDataFrame.loc[:, cols] = sampleDataFrame.loc[:, cols].replace(r'\s*', 0, regex = True)
sampleDataFrame.loc[:, cols] = sampleDataFrame.loc[:, cols].replace(np.nan, 0)

# Creo un file csv a partire dal DataFrame finale
sampleDataFrame.to_csv('dataset/UNSW-NB15_SAMPLING.csv')