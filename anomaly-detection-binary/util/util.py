from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Sample:
    
    def __init__(self, listOfCsv):
        
        # Memorizzo i tipi di attacchi
        self.attack_cat_list_ = ['Backdoors', 'Analysis', 'Fuzzers', 
            'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']

        # Memorizzo tutti i sotto dataframe
        self.df_list_ = []

        # Leggo il primo dataframe
        df = pd.read_csv(listOfCsv[0], encoding = 'utf-8')
        # Trasformo i nomi delle colonne in lettere minuscole
        df.columns = df.columns.str.lower()
        # Elimino tutti gli spazi bianchi prima e dopo l'etichetta
        df['attack_cat'] = df['attack_cat'].str.strip()
        
        # Aggiungo come primo dataframe quello relativo al flusso di rete normale
        self.df_list_.append(df.loc[df['label'] == 0])
        
        # Estraggo il sotto dataframe relativo al flusso di rete anomalo
        df_malicious = df.loc[df['label'] == 1]

        # Sostituisco il valore Backdoor con Backdoors in quanto gli attacchi Backdoors
        # vengono etichettati con queste due stringhe diverse
        df_malicious['attack_cat'].replace(to_replace = 'Backdoor', value = 'Backdoors')

        # Aggiungo alla lista dei dataframe tutti i sotto daframe relativi ad ogni tipologia di attacco
        for attack in self.attack_cat_list_:
            self.df_list_.append(df_malicious.loc[df_malicious['attack_cat'] == attack])

        if int(len(listOfCsv)) > 1:
            # Ripeto questa operazione per ogni altro file csv
            for index_csv in np.arange(1, int(len(listOfCsv))):
                df = pd.read_csv(listOfCsv[index_csv], encoding = 'utf-8')
                df.columns = df.columns.str.lower()
                df['attack_cat'] = df['attack_cat'].str.strip()
                self.df_list_[0] = self.df_list_[0].append(df.loc[df['label'] == 0])
                df_malicious = df.loc[df['label'] == 1]
                df_malicious['attack_cat'].replace(to_replace = 'Backdoor', value = 'Backdoors')
                attack_index = 1
                for attack in self.attack_cat_list_:
                    self.df_list_[attack_index] = self.df_list_[attack_index].append(
                        df_malicious.loc[df_malicious['attack_cat'] == attack])
                    attack_index += 1
        
        # Ordino i dataframe da quello con meno entry a quello con più entry
        self.df_list_.sort(key = lambda d: d.shape[0])

        # Riordino e aggiungo l'etichetta Normal all'array self.attack_cat_list_
        self.attack_cat_list_ = []
        for df in self.df_list_:
            label = df['attack_cat'].values[0]
            if isinstance(label, float):
                self.attack_cat_list_.append('Normal')
            else:
                self.attack_cat_list_.append(label)

        # Creo un unico dataframe
        self.sampleDf_ = pd.DataFrame(columns = self.df_list_[0].columns.values)
        for df in self.df_list_:
            self.sampleDf_ = self.sampleDf_.append(df)

        # Aggiorno lo stato del dataframe
        self.update_status_()

        # Salvo il numero di entry del dataframe più piccolo
        self.minum_ = self.count_attack_cat_list[0]

    def createUniformDataFrame(self, path = None):
        
        # Inizializzo il dataframe
        sampleDf = pd.DataFrame(columns = self.df_list_[0].columns.values)
        
        for i in range(int(len(self.df_list_))):
            df = self.df_list_[i].copy()
            # Le etichette corrispondono agli indici dell'etichetta all'interno dell'array
            # self.attack_cat_list_ incrementato di 1 [1, ..., n]
            # n = |self.attack_cat_list_|
            df['label'] = df['label'].map({1: (i + 1)})
            # Estraggo dal dataframe i-esimo m entry dove m è il numero di entry del dataframe più piccolo
            sampleDf = sampleDf.append(df.sample(n = self.minum_, random_state = 42))

        # Inserisco il valore 0 nelle celle non inizializzate
        cols = sampleDf.columns
        sampleDf.loc[:, cols] = sampleDf.loc[:, cols].replace(np.nan, 0)

        # Esporto il dataframe in un file csv
        if path is not None:
            sampleDf.to_csv(path)

        # Memorizzo il dataframe dopo le operazioni di sampling
        self.sampleDf_ = sampleDf

        # Aggiorno lo stato del dataframe
        self.update_status_()

        return self.sampleDf_.copy()

    def createWeightedDataFrame(self, path = None, weight = 1):
        
        # Inizializzo il dataframe
        sampleDf = pd.DataFrame(columns = self.df_list_[0].columns.values)
        
        for i in range(int(len(self.df_list_))):
            df = self.df_list_[i].copy()
            # Le etichette corrispondono agli indici dell'etichetta all'interno dell'array
            # self.attack_cat_list_ incrementato di 1 [1, ..., n]
            # n = |self.attack_cat_list_|
            df['label'] = df['label'].map({1: (i + 1)})
            # Estraggo dal dataframe i-esimo m entry dove m è:
            # il numero di entry del dataframe più piccolo se i = 0
            # oppure è il weight% più grande rispetto al numero di entry del dataframe (i - 1)-esimo
            if i > 0:
                increase = int(self.minum_ * (1 + weight) ** i)
                if df.shape[0] >= increase:
                    sampleDf = sampleDf.append(df.sample(n = increase, random_state = 42))
                else:
                    sampleDf = sampleDf.append(df)
            else:
                sampleDf = sampleDf.append(df.sample(n = self.minum_))

        # Inserisco il valore 0 nelle celle non inizializzate
        cols = sampleDf.columns
        sampleDf.loc[:, cols] = sampleDf.loc[:, cols].replace(np.nan, 0)

        # Esporto il dataframe in un file csv
        if path is not None:
            sampleDf.to_csv(path)

        # Memorizzo il dataframe dopo le operazioni di sampling
        self.sampleDf_ = sampleDf

        # Aggiorno lo stato del dataframe
        self.update_status_()

        return self.sampleDf_.copy()

    def extractSubDataFrame(self, features, path = None):

        sampleDf = None
        init = False

        for feature in features:
            try:
                if not init:
                    columnValue = self.sampleDf_[feature].values
                    sampleDf = pd.DataFrame(data = columnValue, columns = [feature])
                    init = True
                else:
                    sampleDf[feature] = self.sampleDf_[feature].values
            except KeyError:
                print('Ignore the', feature, 'feature...')

        # Esporto il dataframe in un file csv
        if path is not None:
            sampleDf.to_csv(path)

        # Elimino le colonne con le stringhe
        sampleDf = self.deleteStringColumn(sampleDf)

        return sampleDf

    def deleteStringColumn(self, dataFrame):        
        cols_to_remove = []

        for col in dataFrame.columns:
            try:
                _ = dataFrame[col].astype(float)
            except ValueError:
                cols_to_remove.append(col)
                pass

        df = dataFrame[[col for col in dataFrame.columns if col not in cols_to_remove]]
        return df

    def printDataFrameStatus(self):
        ser = pd.Series(data = self.count_attack_cat_list, index = self.attack_cat_list_)
        print(ser)

    def update_status_(self):
        # Dichiaro l'array contenente il numero di entry per attacco
        self.count_attack_cat_list = []

        for attack in self.attack_cat_list_:
            if attack == 'Normal':
                df = self.sampleDf_.loc[self.sampleDf_['label'] == 0]
                self.count_attack_cat_list.append(df.shape[0])
            else:
                df = self.sampleDf_.loc[self.sampleDf_['attack_cat'] == attack]
                self.count_attack_cat_list.append(df.shape[0])

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