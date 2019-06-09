import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append('../')

import argparse
import numpy as np
import pandas as pd
from keras import layers, optimizers
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from features import *
from nn_utils import get_pr_auc_score, aucroc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_processed_features(df, targets, maxlen):
    targets = [y_train, y_val, y_test]
    sents = []
    for target in targets:
        sents.append(df[target.index])

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sents[0])

    X_train = tokenizer.texts_to_sequences(sents[0])
    X_val = tokenizer.texts_to_sequences(sents[1])
    X_test = tokenizer.texts_to_sequences(sents[2])

    vocab_size = len(tokenizer.word_index) + 1 

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_val, X_test, vocab_size

def get_model(embedding_dim, maxlen):
    inp1 = layers.Input(shape=(X_train_tr.shape[1], ))
    inp2 = layers.Input(shape=(X_train_tr_type.shape[1], ))
    inp3 = layers.Input(shape=(X_train_ben.shape[1], ))


    x = layers.Embedding(input_dim=vs1 , 
                         output_dim=50, 
                         input_length=maxlen)(inp1)
    x = layers.Conv1D(
            filters=64, 
            kernel_size=2, 
            strides=2)(x)
    x = layers.GlobalMaxPool1D()(x)
    # x = layers.MaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.LSTM(10)(x)
    x = layers.Dense(1)(x)


    y = layers.Embedding(input_dim=vs2, 
                         output_dim=50, 
                         input_length=maxlen)(inp2)
    y = layers.Conv1D(
            filters=64,  
            kernel_size=2, 
            strides=2, 
            input_shape=(maxlen, 50))(y)
    y = layers.GlobalMaxPool1D()(y)
    # y = layers.MaxPool1D()(y)
    y = layers.Dropout(0.5)(y)
    # y = layers.LSTM(10)(y)
    y = layers.Dense(1)(y)

    z = layers.Embedding(input_dim=vs3, 
                         output_dim=50, 
                         input_length=maxlen)(inp3)
    z = layers.Conv1D(
            filters=64, 
            kernel_size=2, 
            strides=2, 
            input_shape=(maxlen, 50))(z)
    # z = layers.MaxPool1D()(z)
    z = layers.GlobalMaxPool1D()(z)
    z = layers.Dropout(0.5)(z)
    # z = layers.LSTM(10)(z)
    z = layers.Dense(1)(z)


    w = layers.concatenate([x, y, z])
    w =  layers.Dense(64, activation='relu')(w)
    w =  layers.Dense(128, activation='relu')(w)
    w =  layers.Dense(64, activation='relu')(w)
    w =  layers.Dense(32, activation='relu')(w)
    w = layers.Dropout(0.8)(w)
    out =  layers.Dense(1, activation='sigmoid')(w)

    adam = optimizers.Adam(lr=0.0005, decay=.000005)

    model = Model(inputs=[inp1, inp2, inp3], outputs=out)

    model.compile(loss='binary_crossentropy',
            optimizer=adam,
            metrics=[aucroc])   
    return model

parser = argparse.ArgumentParser(
    description=(
        'This script fit the simple neural network. ' 
        ))

if __name__ == '__main__':
    args = parser.parse_args()

    print('Data preparing... It will start soon!')
    data = pd.read_pickle('data/data.pkl')
    data_treatment = pd.read_pickle('data/treatment.pkl')
    data_treatment_type = pd.read_pickle('data/treatment_type.pkl')
    data_ben_type = pd.read_pickle('data/ben_type.pkl')

    train, val, test = train_val_test_split(data)
    
    targets = []
    for df in [train, val, test]:
        targets.append(get_target(df))
    y_train, y_val, y_test = targets

    X_train_tr, X_val_tr, X_test_tr, vs1 = get_processed_features(
        data_treatment, targets, maxlen=50)

    X_train_tr_type, X_val_tr_type, X_test_tr_type, vs2 = get_processed_features(
        data_treatment_type, targets, maxlen=50)
    
    X_train_ben, X_val_ben, X_test_ben, vs3 = get_processed_features(
        data_ben_type, targets, maxlen=50)

    model = get_model(200, maxlen=50)

    for i in range(15):
        history = model.fit([X_train_tr, X_train_tr_type, X_train_ben], y_train,
                        epochs=5,
                        verbose=True,
                        # shuffle=True,
                        validation_data=([X_val_tr, X_val_tr_type, X_val_ben], y_val),
                        batch_size=3000)
        y_pred = model.predict([X_test_tr, X_test_tr_type, X_test_ben])
        get_pr_auc_score(y_test, y_pred)