import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import logging
import os


def get_measure_accuracy(labels, y_test, y_pred):
    n_labels = int(len(labels))

    # Init matrix
    measures = []
    init_values = []
    for i in range(n_labels):
        init_values.append(0)
    for i in range(n_labels):
        measures.append(init_values.copy())

    n_test = int(len(y_test))
    for i in range(n_test):
        label_expect = y_test[i]
        label_predict = y_pred[i]
        measures[label_expect][label_predict] += 1

    return pd.DataFrame(data=measures, index=labels, columns=labels)


def create_heatmap(df, name_file):
    # Disable logging
    logging.disable(logging.CRITICAL)

    # Create copy of dataframe
    df_copy = df.copy()
    # Convert value in percentage
    df_copy = df_copy.apply(lambda x: round(x / x.sum(), 2), axis=1)

    # Create heatmap
    fig, ax = plt.subplots()
    ax = sns.heatmap(df_copy, annot=True)
    fig.tight_layout()

    # Save fig
    fig.savefig(name_file, dpi=100)
    # Close plot
    plt.close()

    # Enable logging
    logging.disable(logging.NOTSET)

    # Return accuracy for each label
    return np.diag(df_copy)


def create_bar_stacked(df, labels, name_file):
    # Disable logging
    logging.disable(logging.CRITICAL)

    # Create copy of dataframe
    df_copy = df.copy()

    # Save number of attack labeling like normal
    df_error_normal = df_copy['Normal'].values
    df_error_normal = df_error_normal[1:]

    # Delete column and row relative label 'Normal'
    df_copy = df_copy.iloc[1:, 1:]

    # Save number of true positive
    df_true_positive = np.diagonal(df_copy.copy().values)

    # Fill diagonal with zero
    np.fill_diagonal(df_copy.values, 0)

    # Save number of error in attack labeling
    df_error_malicious = df_copy.sum(axis=1).values

    # Index of dataframe
    index = ['False Negative', 'Error in attack labeling', 'True Positive']

    # Create dataframe
    df_copy = pd.DataFrame(data=[df_error_normal, df_error_malicious, df_true_positive],
                           index=index,
                           columns=labels)
    # Transpose dataframe
    df_copy = df_copy.T

    # Convert value in percentage
    df_copy = df_copy.apply(lambda x: round(x / x.sum(), 5), axis=1)

    # Define colors
    colors = ['#D9371E', '#EDED0E', '#15ED0E']

    # Create bar stacked
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fields = df_copy.columns.tolist()
    left = len(df_copy) * [0]
    for idx, name in enumerate(fields):
        plt.barh(df_copy.index, df_copy[name], left=left, color=colors[idx])
        left = left + df_copy[name]
    plt.title('View of errors in labeling', loc='left')
    plt.legend(index, bbox_to_anchor=([0.58, 1, 0, 0]), ncol=4, frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    x_ticks = np.arange(0, 1.1, 0.1)
    x_labels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.xticks(x_ticks, x_labels)
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    fig.tight_layout()

    # Save fig
    fig.savefig(name_file, dpi=100)
    # Close plot
    plt.close()

    # Enable logging
    logging.disable(logging.NOTSET)


def create_grouped_bar(accuracies, weights, labels, name_file):
    # Disable logging
    logging.disable(logging.CRITICAL)

    N_WEIGHTS = int(len(weights))
    N_LABELS = int(len(labels))

    # Get accuracies for each attack
    accuracies_df = np.array(accuracies.copy()).T.tolist()

    # Create group
    for i in range(N_LABELS):
        tmp = [labels[i][0:5]]
        tmp.extend(accuracies_df[i])
        accuracies_df[i] = tmp

    # Create x_ticks
    columns = ['Attack cat']
    columns.extend(weights)

    # Create dataframe to plot
    df = pd.DataFrame(accuracies_df,
                      columns=columns)

    # Create grouped bar
    df.plot(x='Attack cat',
            kind='bar',
            stacked=False,
            title='Accuracy for each labels using the different weights')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=N_WEIGHTS)
    plt.tick_params(axis='x', which='major', labelsize=9)
    plt.xticks(rotation=90)
    plt.xlabel('Weight')
    plt.tight_layout()

    # Save fig
    plt.savefig(name_file, dpi=100)
    # Close plot
    plt.close()

    # Enable logging
    logging.disable(logging.NOTSET)


def log_train_test_status(list_attack_cat, y_train, y_test):
    unique, counts = np.unique(y_train, return_counts=True)
    logging.info('\ny_train status:')
    logging.info(dict(zip(list_attack_cat, counts)))

    unique, counts = np.unique(y_test, return_counts=True)
    logging.info('\ny_pred status:')
    logging.info(dict(zip(list_attack_cat, counts)))


def create_directory(path):
    if path == '':
        return
    try:
        dir_name = os.path.dirname(path)
        create_directory(dir_name)
        if path != '':
            os.mkdir(dir_name)
    except:
        pass
