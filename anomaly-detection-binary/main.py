from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from util.sample import Sample

import numpy as np
import logging
import time
import os

SEPARATOR = '\n------------------------------------\n'

# Inizializzo il logging
logging.basicConfig(filename = os.getcwd() + '\\log\\random-forest.log', 
    format = '%(message)s', level = logging.DEBUG, filemode = 'w')

# Inizializzo l'array list_of_csv con i path assoluti dei file csv che andremo ad utilizzare
list_of_csv = []
for index in np.arange(1, 5):
    list_of_csv.append(os.getcwd() + '\\dataset\\UNSW-NB15_' + str(index) + '.csv')

# Creo un'istanza dell'oggetto Sample e loggo lo stato del dataframe
sample = Sample(list_of_csv)
logging.info('Dataframe status before modifies:')
logging.info(sample.getDataFrameStatus())

for w in [0.25, 0.5, 0.75, 1]:
    # L'array FEATURES contiene tutte le features che andremo ad utilizzare
    FEATURES = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
        'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sintpkt', 'dintpkt',
        'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
        'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'ct_srv_src', 'ct_state_ttl',
        'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
        'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
        'attack_cat', 'label']
    
    # Loggo lo stato di avanzamento
    print('\nStart test with weight ' + str(w) + '...')
    logging.info(SEPARATOR)
    logging.info('Test with weight: ' + str(w) + '\n')

    # Ottengo un sotto dataframe tramite il metodo createWeightedDataFrame
    sample.createWeightedDataFrame(weight = w)
    
    # Ottengo un sotto dataframe utilizzando solo le features contenute nell'array FEATURES
    df = sample.extractSubDataFrame(FEATURES)
    # Visualizza lo stato del DataFrame
    # Coppie (x, y) dove:
    # x è il tipo di attacco
    # y è il numero di entry contenute nel dataframe per quell'attacco
    logging.info('Dataframe status:')
    logging.info(sample.getDataFrameStatus())

    # Assegniamo a FEATURES le features che effettivamente esistono
    FEATURES = df.columns.values
    # Scarto 3 features non numeriche (Proto, Service, State)
    FEATURES = np.append(FEATURES[0], FEATURES[4:-1])
    # Salvo il numero di features
    N_FEATURES = int(len(FEATURES))
    N_FEATURES_DELETED = 3

    # Inizializzo la matrice X
    X = df.iloc[:, np.append([0], np.arange(4, N_FEATURES + N_FEATURES_DELETED))].values

    # Inizializzo il vettore target
    y = df.iloc[:, N_FEATURES + N_FEATURES_DELETED].values

    # Standardizzo i valori delle features
    X = StandardScaler().fit_transform(X)

    # Divido il dataset in 70% traning 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    start_time = time.time()

    # Creiamo un'istanza di RandomForest composta da 100 alberi di decisione
    clf = RandomForestClassifier(n_estimators = 100)

    # Addestriamo l'algoritmo sul dataset di training
    clf.fit(X_train, y_train)

    # Eseguiamo l'algoritmo sul dataset di testing
    y_pred = clf.predict(X_test)

    # Confronto il vettore target con il vettore delle predizioni per misurare e mostrare l'accuracy
    accuracystamp = '\nAccuracy: ' + str(metrics.accuracy_score(y_test, y_pred))
    logging.info(accuracystamp)

    # Stampo il tempo di esecuzione dell'algoritmo
    timestamp = '\nTime elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)'
    logging.info(timestamp)

    print('...Finish test with weight ' + str(w))