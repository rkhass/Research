import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append('../')

import argparse
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from features import *
from nn_utils import get_pr_auc_score, aucroc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_model(maxlen):

    model = Sequential()
    # model.add(layers.Embedding(
    #     input_dim=vocab_size, 
    #     output_dim=embedding_dim, 
    #     input_length=maxlen))
    # model.add(layers.Conv1D(
    #     filters=64, 
    #     kernel_size=2, 
    #     strides=1))
    # model.add(layers.MaxPooling1D())

    # model.add(layers.Conv1D(
    #     filters=64, 
    #     kernel_size=2, 
    #     strides=1))
    # model.add(layers.MaxPooling1D())

    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.LSTM(50))
    # model.add(layers.BatchNormalization())
    # model.add(layers.GlobalMaxPool1D())
    # model.add(layers.Dense(n_neurons, activation='relu'))
    # model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))  
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[aucroc])    
    return model

def get_pad_sequences(df, on, maxlen=70):
    sequences = []

    for name, d in df.groupby('id'):
        sequences.append(d.cost.values)

    pad_seq = pad_sequences(sequences, maxlen=maxlen, dtype='float32')

    pad_seq = pad_seq.reshape(-1, maxlen, 1)
    return pad_seq

parser = argparse.ArgumentParser(
    description=(
        'This script fit the simple neural network. ' 
        ))

if __name__ == '__main__':
    args = parser.parse_args()

    print('\nData preparing... It will start soon!')
    data = pd.read_pickle('data/data.pkl').iloc[:100000]
    scaler = StandardScaler()
    data.cost = scaler.fit_transform(
        np.log(1 + data.cost.values).reshape(-1, 1))
    train, val, test = train_val_test_split(data)
    
    targets = []
    for df in [train, val, test]:
        targets.append(get_target(df).values.reshape(-1, 1))
    y_train, y_val, y_test = targets

    maxlen = 70
    X_train = get_pad_sequences(train, on='cost', maxlen=maxlen)
    X_val = get_pad_sequences(val, on='cost', maxlen=maxlen)
    X_test = get_pad_sequences(test, on='cost', maxlen=maxlen)

    model = get_model(maxlen)

    for i in range(10):
        history = model.fit(X_train, y_train,
                        epochs=5,
                        verbose=True,
                        # shuffle=True,
                        validation_data=(X_val, y_val),
                        batch_size=1024)
        y_pred = model.predict(X_test)
        get_pr_auc_score(y_test, y_pred)