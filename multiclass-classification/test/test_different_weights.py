from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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
NAME_DIR = '01 - Test different weights\\'
IMAGE_PATH = CWD + 'image\\' + NAME_DIR
IMAGE_PATH_BAR_STACKED = IMAGE_PATH + BAR_STACKED
IMAGE_PATH_HEAT_MAP = IMAGE_PATH + HEATMAP
LOG_PATH = CWD + 'log\\' + NAME_DIR + LOGGER
DIRECTORIES = [IMAGE_PATH_BAR_STACKED, IMAGE_PATH_HEAT_MAP, LOG_PATH]

# Attack's label to extract
list_attack = ['Backdoors', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance',
               'Exploits', 'DoS', 'Worms', 'Generic']


def test_different_weights(weights):
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
    sample = Sample(list_csv=list_csv, list_attack=list_attack)
    logging.info('Dataframe status before modifies:')
    logging.info(sample)

    # Init list of accuracy for each weights
    accuracies = []

    for weight in weights:
        # Log test with weight = weight
        print('\nStart test with weight ' + str(weight) + '...')
        logging.info(SEPARATOR)
        logging.info('Test with weight: ' + str(weight) + '\n')

        # Generate sub dataframe with weight = weight
        sample.generate_weighted_df(weight=weight)
        logging.info('Dataframe status:')
        logging.info(sample)

        # Extract sub dataframe and columns of labels
        df, labels = sample.extract_sub_df(features=FEATURES)

        # Init X matrix and vectors y
        X = df.iloc[:, :].values
        y = labels.values

        # Standardize value of matrix
        X = StandardScaler().fit_transform(X)

        # Divide dataset in 70% train and 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # Log status of train test
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
        accuracy = create_heatmap(measure_accuracy,
                                  IMAGE_PATH_HEAT_MAP + 'heatmap-weight' + str(weight) + '.png')
        # Append accuracy for each labels to accuracies array
        accuracies.append(accuracy)

        # Create bar stacked and save fig
        create_bar_stacked(
            measure_accuracy, sample.get_list_attack_cat()[1:],
            IMAGE_PATH_BAR_STACKED + 'bar-stacked-weight' + str(weight) + '.png')

        # Log finish test
        print('...Finish test with weight ' + str(weight))

    # Create grouped bar and save fig
    create_grouped_bar(accuracies, weights, sample.get_list_attack_cat(),
                       IMAGE_PATH + 'grouped-bar.png')

    # Close logger
    logging.shutdown()

if __name__ == '__main__':
    test_different_weights([0.25, 0.5, 0.75, 1])
