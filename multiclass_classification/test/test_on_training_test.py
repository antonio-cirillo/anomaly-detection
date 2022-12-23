from multiclass_classification.test import CWD
from multiclass_classification.test import CONFUSION_MATRIX
from multiclass_classification.test import DETECTION_RATE
from multiclass_classification.test import LOGGER
from multiclass_classification.test import SEPARATOR

from multiclass_classification.src.util import plot_confusion_matrix
from multiclass_classification.src.util import plot_recall
from multiclass_classification.src.util import log_train_test_status
from multiclass_classification.src.util import create_directory

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import logging
import pickle
import os

# Declare all path
NAME_DIR = '01 - Test on training test'
MODEL_PATH = os.path.join(CWD, 'model', NAME_DIR)
IMAGE_PATH = os.path.join(CWD, 'image', NAME_DIR)
CONFUSION_MATRIX_PATH = os.path.join(IMAGE_PATH, CONFUSION_MATRIX)
DETECTION_RATE_PATH = os.path.join(IMAGE_PATH, DETECTION_RATE)
LOG_BASE = os.path.join(CWD, 'log', NAME_DIR)
LOG_PATH = os.path.join(LOG_BASE, LOGGER)
RESULT_PATH = os.path.join(LOG_BASE, 'results.csv')
DIRECTORIES = [MODEL_PATH, LOG_BASE, CONFUSION_MATRIX_PATH, DETECTION_RATE_PATH]

# Mapping
MAPPING = {
    "Normal": 0,
    "Worms": 1,
    "Backdoors": 2,
    "Shellcode": 3,
    "Analysis": 4,
    "Reconnaissance": 5,
    "DoS": 6,
    "Fuzzers": 7,
    "Exploits": 8,
    "Generic": 9
}

COLUMNS_RESULTS = ['models', 'n_estimators', 'max_depth', 'accuracy_train', 'weighted_accuracy_train',
                   'accuracy_test', 'weighted_accuracy_test', 'macro_precision_avg', 'weighted_precision_avg',
                   'macro_recall_avg', 'weighted_recall_avg', 'macro_f1_score_avg', 'weighted_f1_score_avg']


def __delete_string_columns__(df):
    cols_to_remove = []

    for col in df.columns:
        try:
            _ = df[col].astype(float)
        except ValueError:
            cols_to_remove.append(col)
            pass

    df_ = df[[col for col in df.columns if col not in cols_to_remove]]
    return df_


def test_on_training_test(dataset_path):
    # init directory
    for directory in DIRECTORIES:
        create_directory(directory)
    # init logger
    logging.basicConfig(filename=LOG_PATH, format='%(message)s',
                        level=logging.DEBUG, filemode='w')

    # create path of training and test dataset
    _file_name_training = 'UNSW_NB15_testing-set.csv'
    _file_path_training = os.path.join(dataset_path, _file_name_training)
    _file_name_testing = 'UNSW_NB15_training-set.csv'
    _file_path_testing = os.path.join(dataset_path, _file_name_testing)

    # read dataframe
    df_training = pd.read_csv(_file_path_training, low_memory=False)
    df_testing = pd.read_csv(_file_path_testing, low_memory=False)

    # remove id
    df_training = df_training.iloc[:, 1:]
    df_testing = df_testing.iloc[:, 1:]

    # create df array
    dfs = [df_training, df_testing]

    # clean data
    for i in range(0, len(dfs)):
        dfs[i]['attack_cat'] = dfs[i]['attack_cat'].str.strip()
        # replace NaN with Normal
        dfs[i]['attack_cat'] = dfs[i]['attack_cat'].replace(np.nan, 'Normal')
        # fix bug on labels Backdoor
        dfs[i]['attack_cat'] = dfs[i]['attack_cat'].replace('Backdoor', 'Backdoors')
        # replace label 1 in {0, 1, 2, ..., 9}
        dfs[i]['label'] = dfs[i]['attack_cat'].map(MAPPING)
        dfs[i] = __delete_string_columns__(dfs[i])
        dfs[i] = dfs[i].fillna(0)

    # get x and y of training and test
    x_train = dfs[0].iloc[:, :-1].values
    y_train = dfs[0].iloc[:, -1].values
    x_test = dfs[1].iloc[:, :-1].values
    y_test = dfs[1].iloc[:, -1].values
    logging.info("Dataset status")
    log_train_test_status(MAPPING.keys(), y_train, y_test)

    # create result dataframe
    df = pd.DataFrame(columns=COLUMNS_RESULTS)

    # use different model on different max_depth
    models = [xgb.XGBClassifier, xgb.XGBRFClassifier, RandomForestClassifier, GradientBoostingClassifier]
    n_estimators_ = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    max_depths = [3, 4, 5]
    for model in models:
        model_name = model.__name__
        if model == xgb.XGBRFClassifier or model == xgb.XGBClassifier:
            params = {"model": str(model_name)}
            model_path = os.path.join(MODEL_PATH, f'{model_name}.sav')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as model_file:
                    clf = pickle.load(model_file)
                    print(f'Loaded model: {params}')
            else:
                print(f'Creating model: {params}')
                clf = model()
                clf.fit(x_train, y_train)
                with open(model_path, 'wb') as model_file:
                    pickle.dump(clf, model_file)
                print(f'Saved model: {model_path}')

            # predict on training
            y_pred_train = clf.predict(x_train)
            # predict on test
            y_pred_test = clf.predict(x_test)

            # get result
            accuracy_score_train = metrics.accuracy_score(y_train, y_pred_train)
            w_accuracy_score_train = metrics.balanced_accuracy_score(y_train, y_pred_train)
            accuracy_score_test = metrics.accuracy_score(y_test, y_pred_test)
            w_accuracy_score_test = metrics.balanced_accuracy_score(y_test, y_pred_test)
            m_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=1)
            w_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred_test, average=None)
            m_recall = recall_score(y_test, y_pred_test, average='macro')
            w_recall = recall_score(y_test, y_pred_test, average='weighted')
            m_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=1)
            w_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=1)

            # append row to dataframe
            row = [model_name, '-', '-', accuracy_score_train, w_accuracy_score_train, accuracy_score_test,
                   w_accuracy_score_test, m_precision, w_precision, m_recall, w_recall, m_f1, w_f1]
            df.loc[len(df.index)] = row

            # log results
            logging.info(SEPARATOR)
            logging.info(f'{params}')
            logging.info(f'Accuracy on train: {accuracy_score_train}')
            logging.info(f'Balanced accuracy on train: {w_accuracy_score_train}')
            logging.info(f'Accuracy on test: {accuracy_score_test}')
            logging.info(f'Balanced accuracy on test: {w_accuracy_score_test}')
            logging.info(f'\n{metrics.classification_report(y_test, y_pred_test, digits=3)}')

            # plot confusion matrix
            confusion_matrix_path = os.path.join(CONFUSION_MATRIX_PATH, f'{model_name}.jpeg')
            cm = metrics.confusion_matrix(y_test, y_pred_test)
            plot_confusion_matrix(cm, MAPPING.keys(), confusion_matrix_path)

            # plot detection rate
            detection_rate_path = os.path.join(DETECTION_RATE_PATH, f'{model_name}.jpeg')
            plot_recall(MAPPING.keys(), recall, detection_rate_path)

        else:
            for max_depth in max_depths:
                for n_estimators in n_estimators_:
                    params = {"model": str(model_name), "n_estimators": n_estimators, "max_depth": max_depth}
                    base_name = f'{model_name}-{n_estimators}_max_depth{max_depth}'
                    model_path = os.path.join(MODEL_PATH, f'{base_name}.sav')

                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as model_file:
                            clf = pickle.load(model_file)
                            print(f'Loaded model: {params}')
                    else:
                        print(f'Creating model: {params}')
                        clf = model(n_estimators=n_estimators, max_depth=max_depth)
                        clf.fit(x_train, y_train)
                        with open(model_path, 'wb') as model_file:
                            pickle.dump(clf, model_file)
                        print(f'Saved model: {model_path}')

                    # predict on training
                    y_pred_train = clf.predict(x_train)
                    # predict on test
                    y_pred_test = clf.predict(x_test)

                    # get result
                    accuracy_score_train = metrics.accuracy_score(y_train, y_pred_train)
                    w_accuracy_score_train = metrics.balanced_accuracy_score(y_train, y_pred_train)
                    accuracy_score_test = metrics.accuracy_score(y_test, y_pred_test)
                    w_accuracy_score_test = metrics.balanced_accuracy_score(y_test, y_pred_test)
                    m_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=1)
                    w_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
                    recall = recall_score(y_test, y_pred_test, average=None)
                    m_recall = recall_score(y_test, y_pred_test, average='macro')
                    w_recall = recall_score(y_test, y_pred_test, average='weighted')
                    m_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=1)
                    w_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=1)

                    # append row to dataframe
                    row = [model_name, n_estimators, max_depth, accuracy_score_train, w_accuracy_score_train,
                           accuracy_score_test,
                           w_accuracy_score_test, m_precision, w_precision, m_recall, w_recall, m_f1, w_f1]
                    df.loc[len(df.index)] = row

                    # log results
                    logging.info(SEPARATOR)
                    logging.info(f'{params}')
                    logging.info(f'Accuracy on train: {accuracy_score_train}')
                    logging.info(f'Balanced accuracy on train: {w_accuracy_score_train}')
                    logging.info(f'Accuracy on test: {accuracy_score_test}')
                    logging.info(f'Balanced accuracy on test: {w_accuracy_score_test}')
                    logging.info(f'\n{metrics.classification_report(y_test, y_pred_test, digits=3)}')

                    # plot confusion matrix
                    confusion_matrix_path = os.path.join(CONFUSION_MATRIX_PATH, f'{base_name}.jpeg')
                    cm = metrics.confusion_matrix(y_test, y_pred_test)
                    plot_confusion_matrix(cm, MAPPING.keys(), confusion_matrix_path)

                    # plot detection rate
                    detection_rate_path = os.path.join(DETECTION_RATE_PATH, f'{base_name}.jpeg')
                    plot_recall(MAPPING.keys(), recall, detection_rate_path)

    df.to_csv(RESULT_PATH)

    # close logger
    logging.shutdown()
