from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def minimizeDataFrameByFeatureImportances(classifier, features, path, min):

    feature_imp = pd.Series(classifier.feature_importances_, 
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