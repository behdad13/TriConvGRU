import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array

def split_sequences(sequences, n_steps, for_hor):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - for_hor:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:end_ix + for_hor, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def prepare_datasets(data, train_size_ratio=0.8, valid_size_ratio=0.2, n_timesteps=72, for_hor=3):
    data = pd.read_csv(data, index_col=None)
    data.pop("Unnamed: 0")

    train_size = int(len(data) * train_size_ratio)
    valid_size = int(len(data) * valid_size_ratio)

    df_train, df_val = data[:train_size], data[train_size:]
    
    global scaler

    scaler = MinMaxScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train), index=df_train.index, columns=df_train.columns)
    df_val = pd.DataFrame(scaler.transform(df_val), index=df_val.index, columns=df_val.columns)

    df_train = df_train.to_numpy()
    df_val = df_val.to_numpy()

    X_train, y_train = split_sequences(df_train, n_timesteps, for_hor)
    X_val, y_val = split_sequences(df_val, n_timesteps, for_hor)

    return X_train, y_train, X_val, y_val


def prepare_test_datasets(data, n_timesteps=72, for_hor=3):
    data = pd.read_csv(data, index_col=None)
    data.pop("Unnamed: 0")

    df_test = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    df_test = df_test.to_numpy()

    X_test, y_test = split_sequences(df_test, n_timesteps, for_hor)

    return X_test, y_test

def descale_maker():
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
    return descaler

def descale(values):
    values_2d = np.array(values)[:, np.newaxis]
    return descale_maker().inverse_transform(values_2d).flatten()

