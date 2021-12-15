from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def measureAccuracy(labels = [], y_pred = None, y_test = None):
    n_labels = int(len(labels))

    # Inizializzo l'array di misurazioni
    measures = []
    array = []
    for j in range(n_labels):
        array.append(0)
    for i in range(n_labels):
        measures.append(array.copy())

    n_entries = int(len(y_test))
    for i in range(n_entries):
        label_expect = y_test[i]
        label_predict = y_pred[i]
        measures[label_expect][label_predict] += 1
        array[label_predict] += 1

    df = pd.DataFrame(data = measures,
        index = labels,
        columns = labels)

    print(df)

    df_copy = df.copy()
    df = df.apply(lambda x: round(x / x.sum(), 2), axis=1)

    df_normal = df_copy['Normal'].values
    df_normal = df_normal[1:]
    
    np.fill_diagonal(df_copy.values, 0)

    df_copy = df_copy.iloc[1:, :]
    df_copy = df_copy.iloc[:, 1:]
    df_anomarl = df_copy.sum(axis = 1).values

    index = ['False Negative', 'Error in attack labeling']

    df_copy = pd.DataFrame(data = [df_normal, df_anomarl],
        index = index,
        columns = labels[1:])
    df_copy = df_copy.T

    df_copy = df_copy.apply(lambda x: round(x / x.sum(), 2), axis=1)

    colors = ['#1D2F6F', '#8390FA']
    plot_stackedbar_p(df_copy, index, colors, 'Plot of errors', '')

    return df

def displayHeatMap(metrics, name_file):
    fig, ax = plt.subplots()
    ax = sns.heatmap(metrics, annot = True)
    
    fig.tight_layout()
    fig.savefig(name_file, dpi = 100)
    plt.close()

def deleteStringColumn(df):
    cols_to_remove = []

    for col in df.columns:
        try:
         _ = df[col].astype(float)
        except ValueError:
            cols_to_remove.append(col)
            pass

    # keep only the columns in df that do not contain string
    df = df[[col for col in df.columns if col not in cols_to_remove]]
    return df

def plotFeatureImportances(classifier, features, name_file, n_elements = 5):

    feature_imp = pd.Series(classifier.feature_importances_, 
        index = features).sort_values(ascending = False)

    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis = 0)

    fig, ax = plt.subplots()

    feature_imp[0 : n_elements + 1].plot.bar(yerr = std[0 : n_elements + 1], ax = ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")

    fig.tight_layout()

    fig.savefig(name_file, dpi = 100)
    plt.show()

def minimizeDataFrameByFeatureImportances(feature_imp, features, path, min):
    
    feature_imp = pd.Series(feature_imp, 
        index = features).sort_values(ascending = False)

    dataFrame = pd.read_csv(path, encoding = 'utf-8')
    returnDF = None
    count = 0

    for feature, impurity in feature_imp.items():
        if count == 0:
            returnDF = dataFrame[feature]
            count = impurity
        elif count < min:
            returnDF = pd.concat([returnDF, dataFrame[feature]], axis = 1)
            count += impurity
        else:
            break

    returnDF.to_csv(path.replace('.csv', '_REDUCED.csv'))

    return returnDF

def plot_stackedbar_p(df, labels, colors, title, subtitle):
    fields = df.columns.tolist()
    
    fig, ax = plt.subplots(1, figsize=(8, 6))

    left = len(df) * [0]
    for idx, name in enumerate(fields):
        plt.barh(df.index, df[name], left = left, color=colors[idx])
        left = left + df[name]

    plt.title(title, loc='left')
    plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle)

    plt.legend(labels, bbox_to_anchor=([0.58, 1, 0, 0]), ncol=4, frameon=False)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    xticks = np.arange(0,1.1,0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0,101,10)]
    plt.xticks(xticks, xlabels)

    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    fig.tight_layout()
    
    plt.show()