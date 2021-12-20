from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler

from util.sample import Sample
from util.util import measureAccuracy
from util.util import displayHeatMap
from util.util import displayBarStacked
from util.util import displayGroupedBar
from util.util import logStatusTrainTestSplit
from util.util import createDirectory

import numpy as np
import logging
import time
import os

# Dichiarazione delle costanti e variabili globali
SEPARATOR = '\n------------------------------------\n'
NAME_DIR = '\\03 - Test with random over sampler\\'
CWD = os.getcwd() + '\\'
BAR_STACKED = 'Bar Stacked\\'
HEAT_MAP = 'Heat Map\\'
ATTACK_CAT = ['Normal', 'Backdoors', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance', 
    'Exploits', 'DoS', 'Worms', 'Generic']
accuracies = []

# Dichiaro i path per le immagini e il log
IMAGE_PATH = CWD + 'image' + NAME_DIR
IMAGE_PATH_BAR_STACKED = IMAGE_PATH + BAR_STACKED 
IMAGE_PATH_HEAT_MAP = IMAGE_PATH + HEAT_MAP
PATH_FILE_LOG = CWD + 'log' + NAME_DIR + '\\logger.log'

# Insieme delle cartelle che conterranno le risorse dell'esecuzione
DIRECTORY = ['image\\', 'log\\', 'image' + NAME_DIR, 'log' + NAME_DIR,
    'image' + NAME_DIR + BAR_STACKED, 'image' + NAME_DIR + HEAT_MAP]

def test_with_random_over_sampler(weights):

    # Creo le cartelle se non esistono
    for dir in DIRECTORY:
        createDirectory(CWD + dir)

    # Inizializzo il logging
    logging.basicConfig(filename = PATH_FILE_LOG, 
        format = '%(message)s', level = logging.DEBUG, filemode = 'w')

    # Inizializzo l'array list_of_csv con i path assoluti dei file csv che andremo ad utilizzare
    list_of_csv = []
    for index in np.arange(1, 5):
        list_of_csv.append(CWD + 'dataset\\UNSW-NB15_' + str(index) + '.csv')

    # Creo un'istanza dell'oggetto Sample e loggo lo stato del dataframe
    sample = Sample(list_of_csv)
    logging.info('Dataframe status before modifies:')
    logging.info(sample.getDataFrameStatus())

    for w in weights:
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

        # Logghiamo lo stato del test
        logging.info("\nTrainTest status before augmentation:\n")
        logStatusTrainTestSplit(sample.getLabelsList(), X_train, X_test, y_train, y_test)

        # Facciamo augmentation portando tutti i dati allo stesso stato
        ros = RandomOverSampler()
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

        # Logghiamo lo stato del test
        logging.info("\nTrainTest status after augmentation:\n")
        logStatusTrainTestSplit(sample.getLabelsList(), X_train_res, X_test, y_train_res, y_test)

        # Conto il tempo per addestrare la random forest
        start_time = time.time()

        # Creiamo un'istanza di RandomForest composta da 100 alberi di decisione
        clf = RandomForestClassifier(n_estimators = 100)

        # Addestriamo l'algoritmo sul dataset di training
        clf.fit(X_train_res, y_train_res)

        # Eseguiamo l'algoritmo sul dataset di testing
        y_pred = clf.predict(X_test)

        # Stampo il tempo di esecuzione dell'algoritmo
        timestamp = '\nTime elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)'
        logging.info(timestamp)

        # Confronto il vettore target con il vettore delle predizioni per misurare e mostrare l'accuracy
        accuracystamp = '\nAccuracy: ' + str(metrics.accuracy_score(y_test, y_pred)) + '\n'
        logging.info(accuracystamp)

        # Ottengo il dataset di misurazione e lo loggo
        measures = measureAccuracy(labels = sample.getLabelsList(), y_test = y_test, y_pred = y_pred)
        logging.info(measures)
        
        # Creo l'heatmap e salvo l'accuracy relativa ad ogni attack_cat
        accuracy_attack_cat = displayHeatMap(
            measures, IMAGE_PATH_HEAT_MAP + 'heat-map-weight' + str(w) + '.png')
        accuracies.append(accuracy_attack_cat)

        # Creo il barchart stacked
        colors = ['#1D2F6F', '#8390FA']
        displayBarStacked(measures, sample.getLabelsList()[1:], colors, 'Errors stacked',
            IMAGE_PATH_BAR_STACKED + 'bar-stacked' + str(w) + '.png')

        print('...Finish test with weight ' + str(w))

    # Mostro l'accuracy per le diverse etichette al cambiamento dei pesi
    displayGroupedBar(weights, accuracies, sample.getLabelsList(),
        IMAGE_PATH + 'grouped-bar.png')

if __name__ == '__main__':
    test_with_random_over_sampler([0.25, 0.5, 0.75, 1])