import pandas as pd
import numpy as np
import os

NAME_FEATURES_FILE = 'NUSW-NB15_features.csv'
BASE_NAME_CSV_FILE = 'UNSW-NB15_'


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


class Sample:

    def __init__(self, path_directory: str):

        self._path_directory = path_directory
        self._path_features_file = os.path.join(path_directory, NAME_FEATURES_FILE)
        self.features = []
        self.df_list_ = []
        self.labels_ = []
        self.list_attack_ = []

        # read features
        df = pd.read_csv(self._path_features_file)
        self.features = df['Name'].values

        # create dataframe
        df = pd.DataFrame()

        # read csv file
        for i in range(1, 5):
            _file_name = f'{BASE_NAME_CSV_FILE}{i}.csv'
            _file_path = os.path.join(path_directory, _file_name)
            _df = pd.read_csv(_file_path, header=None, low_memory=False)
            _df.columns = self.features
            df = pd.concat([df, _df])

        # delete whitespace
        df['attack_cat'] = df['attack_cat'].str.strip()
        # replace NaN with Normal
        df['attack_cat'] = df['attack_cat'].replace(np.nan, 'Normal')
        # fix bug on labels Backdoor
        df['attack_cat'] = df['attack_cat'].replace('Backdoor', 'Backdoors')

        # first dataset is relative normal network
        self.df_list_.append(df.loc[df['Label'] == 0])

        # extract dataset relative malicious network
        df_malicious = df.loc[df['Label'] == 1]
        self.list_attack_ = np.unique(df_malicious['attack_cat'].values)

        # extract sub-dataset relative every label contains in labels and adding to df_list
        for attack in self.list_attack_:
            self.df_list_.append(df_malicious.loc[df_malicious['attack_cat'] == attack])

        # order list of dataset by entry
        self.df_list_.sort(key=lambda d: d.shape[0])
        # order labels by entry
        self.labels_ = []
        for _df in self.df_list_:
            label = _df['attack_cat'].values[0]
            self.labels_.append(label)

        # create local dataframe
        self.df_sample_ = pd.DataFrame(columns=self.features)
        for _df in self.df_list_:
            self.df_sample_ = pd.concat([self.df_sample_, _df])

        # insert 0 inside parameter not init
        self.df_sample_.loc[:, :] = self.df_sample_.loc[:, :].replace(np.nan, 0)
        # update dataframe status
        self.__update_status__()

        # Save row number of minim dataframe
        self.min_ = self.df_list_[0].shape[0]

    def generate_weighted_df(self, weight=1, path=None):
        # init dataframe
        df_sample_ = pd.DataFrame(columns=self.features)

        # generate weighted dataframe
        n_df = int(len(self.df_list_))
        for i in range(n_df):
            df = self.df_list_[i].copy()
            df['Label'] = df['Label'].map({1: (i + 1), 0: 0})
            if i > 0:
                increment = int(self.min_ * (1 + weight) ** i)
                if df.shape[0] >= increment:
                    df_sample_ = pd.concat([df_sample_, df.sample(n=increment)])
                else:
                    df_sample_ = pd.concat([df_sample_, df])
            else:
                df_sample_ = pd.concat([df_sample_, df])

        # export dataframe if path is init
        if path is not None:
            df_sample_.to_csv(path)

        # save new dataframe inside class variable
        self.df_sample_ = df_sample_
        # update dataframe status
        self.__update_status__()

        # return copy of dataframe
        return self.df_sample_.copy()

    def extract_sub_df(self, path=None):
        # init sub_df
        sub_df = None
        init = False

        # save vectors of labels
        y = self.df_sample_['Label'].copy()

        # extract column contained inside features
        for feature in self.features:
            try:
                if not init:
                    column = self.df_sample_[feature].values
                    sub_df = pd.DataFrame(data=column, columns=[feature])
                    init = True
                else:
                    sub_df[feature] = self.df_sample_[feature].values
            except KeyError:
                print('Ignore the', feature, 'features...')

        # export dataframe if path is init
        if path is not None:
            sub_df.to_csv(path)

        # delete columns contains string value
        sub_df = __delete_string_columns__(sub_df)
        # insert 0 inside parameter not init
        sub_df.iloc[:, :] = sub_df.iloc[:, :].replace(np.nan, 0)

        # return sub dataframe without and labels
        return sub_df, y

    def get_list_attack_cat(self):
        return np.roll(self.labels_.copy(), 1)

    def __update_status__(self):
        # init counter
        self.list_n_entry = []
        # count entry for all dataframe
        for label in self.labels_:
            self.list_n_entry.append(
                self.df_sample_.loc[self.df_sample_['attack_cat'] == label].shape[0])

    def __str__(self):
        return str(pd.Series(data=self.list_n_entry, index=self.labels_))
