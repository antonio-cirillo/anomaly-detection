import pandas as pd
import numpy as np


class NoFilePassed(Exception):
    pass


class CantMergeOneLabel(Exception):
    pass


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

    def __init__(self, list_csv=[], list_attack=[], labels_to_merge=[]):

        self.labels_ = []
        self.df_list_ = []
        self.df_sample_ = None
        self.list_n_entry = []
        self.min_ = 0

        # Check number of csv
        n_csv = int(len(list_csv))
        if n_csv == 0:
            raise NoFilePassed('list_csv must be have more than 0 element')

        # Init dataset with first file csv
        df = pd.read_csv(list_csv[0], encoding='utf-8')
        # To lower features
        df.columns = df.columns.str.lower()
        # Delete whitespace
        df['attack_cat'] = df['attack_cat'].str.strip()
        # Replace NaN with Normal
        df['attack_cat'] = df['attack_cat'].replace(np.nan, 'Normal')

        # First dataset is relative normal network
        self.df_list_.append(df.loc[df['label'] == 0])
        self.labels_.append('Normal')

        # Extract dataset relative malicious network
        df_malicious = df.loc[df['label'] == 1]
        # Fix bug on labels Backdoor
        df_malicious['attack_cat'].replace(to_replace='Backdoor', value='Backdoors')

        # Extract sub-dataset relative every label contains in labels and adding to df_list
        for attack in list_attack:
            self.df_list_.append(df_malicious.loc[df_malicious['attack_cat'] == attack])
            self.labels_.append(attack)

        # Repeat this operation if list_of_csv contains more than one file
        if n_csv > 1:
            for i in range(1, n_csv):
                df = pd.read_csv(list_csv[i], encoding='utf-8')
                df.columns = df.columns.str.lower()
                df['attack_cat'] = df['attack_cat'].str.strip()
                df['attack_cat'] = df['attack_cat'].replace(np.nan, 'Normal')
                self.df_list_[0] = self.df_list_[0].append(df.loc[df['label'] == 0])
                df_malicious = df.loc[df['label'] == 1]
                df_malicious['attack_cat'].replace(to_replace='Backdoor', value='Backdoors')
                j = 1
                for label in list_attack:
                    self.df_list_[j] = self.df_list_[j].append(
                        df_malicious.loc[df_malicious['attack_cat'] == label])
                    j += 1

        # Merge labels contains in labels_to_merge
        n_labels_merge = int(len(labels_to_merge))
        if n_labels_merge > 0:
            if n_labels_merge == 1:
                raise CantMergeOneLabel("Can't merge one label")
            else:
                label_merge = labels_to_merge[0]
                index_label = self.labels_.index(label_merge)
                self.df_list_[index_label]['attack_cat'] = 'Merge'
                self.labels_[index_label] = 'Merge'
                for i in range(1, n_labels_merge):
                    label_merge = labels_to_merge[i]
                    index_label_to_add = self.labels_.index(label_merge)
                    self.df_list_[index_label_to_add]['attack_cat'] = 'Merge'
                    self.df_list_[index_label] = self.df_list_[index_label].append(
                        self.df_list_[index_label_to_add])
                    del self.df_list_[index_label_to_add]
                    del self.labels_[index_label_to_add]

        # Order list of dataset by entry
        self.df_list_.sort(key=lambda d: d.shape[0])

        self.labels_ = []
        # Order labels by entry
        for df in self.df_list_:
            label = df['attack_cat'].values[0]
            self.labels_.append(label)

        # Create local dataframe
        self.df_sample_ = pd.DataFrame(columns=self.df_list_[0].columns.values)
        for df in self.df_list_:
            self.df_sample_ = self.df_sample_.append(df)

        # Insert 0 inside parameter not init
        self.df_sample_.loc[:, :] = self.df_sample_.loc[:, :].replace(np.nan, 0)

        # Update dataframe status
        self.__update_status__()

        # Save row number of minim dataframe
        self.min_ = self.df_list_[0].shape[0]

    def __update_status__(self):
        # Init counter
        self.list_n_entry = []

        # Count entry for all dataframe
        for label in self.labels_:
            self.list_n_entry.append(
                self.df_sample_.loc[self.df_sample_['attack_cat'] == label].shape[0])

    def generate_weighted_df(self, weight=1, path=None):
        # Init dataframe
        df_sample_ = pd.DataFrame(columns=self.df_list_[0].columns.values)

        # Generate new df
        n_df = int(len(self.df_list_))
        for i in range(n_df):
            df = self.df_list_[i].copy()
            df['label'] = df['label'].map({1: (i + 1)})
            if i > 0:
                increment = int(self.min_ * (1 + weight) ** i)
                if df.shape[0] >= increment:
                    df_sample_ = df_sample_.append(df.sample(n=increment))
                else:
                    df_sample_ = df_sample_.append(df)
            else:
                df_sample_ = df_sample_.append(df)

        # Insert 0 inside parameter not init
        df_sample_.loc[:, :] = df_sample_.loc[:, :].replace(np.nan, 0)

        # Export dataframe if path is init
        if path is not None:
            df_sample_.to_csv(path)

        # Save new dataframe inside class variable
        self.df_sample_ = df_sample_

        # Update dataframe status
        self.__update_status__()

        # Return copy of dataframe
        return self.df_sample_.copy()

    def extract_sub_df(self, features, path=None):
        # Init sub_df
        sub_df = None
        init = False

        # Save vectors of labels
        y = self.df_sample_['label'].copy()

        # Extract column contained inside features
        for feature in features:
            try:
                if not init:
                    column = self.df_sample_[feature].values
                    sub_df = pd.DataFrame(data=column, columns=[feature])
                    init = True
                else:
                    sub_df[feature] = self.df_sample_[feature].values
            except KeyError:
                print('Ignore the', feature, 'features...')

        # Export dataframe if path is init
        if path is not None:
            sub_df.to_csv(path)

        # Delete columns contains string value
        sub_df = __delete_string_columns__(sub_df)

        # Return sub dataframe without and labels
        return sub_df, y

    def get_df(self):
        return self.df_sample_.copy()

    def get_list_attack_cat(self):
        return np.roll(self.labels_.copy(), 1)

    def __str__(self):
        return str(pd.Series(data=self.list_n_entry, index=self.labels_))
