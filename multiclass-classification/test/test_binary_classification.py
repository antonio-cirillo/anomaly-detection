from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from imblearn.over_sampling import SMOTE

from src.sample import Sample
from src.util import get_measure_accuracy
from src.util import create_heatmap
from src.util import create_bar_stacked
from src.util import create_grouped_bar
from src.util import create_directory
from src.util import log_train_test_status

from test import SEPARATOR
from test import CWD
from test import BAR_STACKED
from test import HEATMAP
from test import LOGGER
from test import FEATURES

import logging
import time

# Declare all path
NAME_DIR = '04 - Test binary classification\\'
IMAGE_PATH = CWD + 'image\\' + NAME_DIR
IMAGE_PATH_HEAT_MAP = IMAGE_PATH + HEATMAP
LOG_PATH = CWD + 'log\\' + NAME_DIR + LOGGER
DIRECTORIES = [IMAGE_PATH_HEAT_MAP, LOG_PATH]

# Attack's label to extract
list_attack = ['Backdoors', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance',
               'Exploits', 'DoS', 'Worms', 'Generic']


def test_binary_classification():
    # Create folder
    for directory in DIRECTORIES:
        create_directory(directory)

    # Init logger
    logging.basicConfig(filename=LOG_PATH, format='%(message)s',
                        level=logging.DEBUG, filemode='w')

    # Init list csv
    list_csv = []
    for i in range(1, 5):
        list_csv.append(CWD + 'dataset\\UNSW-NB15_' + str(i) + '.csv')

    # Create sample object and log status
    sample = Sample(list_csv=list_csv, list_attack=list_attack,
                    labels_to_merge=list_attack)
    logging.info('Dataframe status:')
    logging.info(sample)

    # Log test start
    print('\nStart test...')

    # Extract sub dataframe and columns of labels
    df, labels = sample.extract_sub_df(features=FEATURES)

    # Init X matrix and vectors y
    X = df.iloc[:, :].values
    y = labels.values
    y = y.astype('int')

    # Standardize value of matrix
    X = StandardScaler().fit_transform(X)

    # Divide dataset in 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Log status of train test before data augmentation
    log_train_test_status(sample.get_list_attack_cat(), y_train, y_test)

    # Create random forest classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Start timing
    start_time = time.time()

    # Fit random forest
    clf.fit(X_train, y_train)

    # Test random forest
    y_pred = clf.predict(X_test)

    # Stop timing and log it
    logging.info('\nTime elapsed: ' + str(round(time.time() - start_time, 2)) + ' second(s)')

    # Get accuracy and log it
    logging.info('\nAccuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))

    # Get matrix of accuracy
    measure_accuracy = get_measure_accuracy(
        sample.get_list_attack_cat(), y_test, y_pred)

    # Create heatmap and save fig
    create_heatmap(measure_accuracy,
                   IMAGE_PATH_HEAT_MAP + 'heatmap-weight.png')

    # Log finish test
    print('...Finish test')


# Close logger
logging.shutdown()

if __name__ == '__main__':
    test_binary_classification()
