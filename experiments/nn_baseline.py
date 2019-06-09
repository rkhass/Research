import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append('../')

import argparse
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from features import *
from nn_utils import get_pr_auc_score, aucroc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_model(embedding_dim, vocab_size, maxlen, n_neurons):

    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=maxlen))
    model.add(layers.Conv1D(
        filters=64, 
        kernel_size=3, 
        strides=1, 
        input_shape=(maxlen, embedding_dim)))
    # model.add(layers.BatchNormalization())
    model.add(layers.GlobalMaxPool1D())
    # model.add(layers.Dense(n_neurons, activation='relu'))
    model.add(layers.Dense(n_neurons, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[aucroc])    
    return model

parser = argparse.ArgumentParser(
    description=(
        'This script fit the simple neural network. ' 
        ))

if __name__ == '__main__':
    args = parser.parse_args()

    print('\nData preparing... It will start soon!')
    data = pd.read_pickle('data/data.pkl')
    data_treatment = pd.read_pickle('data/treatment.pkl')
    train, val, test = train_val_test_split(data)
    
    targets = []
    for df in [train, val, test]:
        targets.append(get_target(df))
    y_train, y_val, y_test = targets

    frames = [y_train, y_val, y_test]
    sents = []
    for df in frames:
        sents.append(data_treatment[df.index])

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sents[0])

    X_train = tokenizer.texts_to_sequences(sents[0])
    X_val = tokenizer.texts_to_sequences(sents[1])
    X_test = tokenizer.texts_to_sequences(sents[2])

    vocab_size = len(tokenizer.word_index) + 1 
    print(vocab_size)

    maxlen = 100
    X_train = pad_sequences(X_train, padding='pre', maxlen=maxlen)
    X_val = pad_sequences(X_val, padding='pre', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='pre', maxlen=maxlen)
    model = get_model(100, vocab_size, maxlen, 10)

    for i in range(10):
        history = model.fit(X_train, y_train,
                        epochs=5,
                        verbose=True,
                        # shuffle=True,
                        validation_data=(X_val, y_val),
                        batch_size=15000)
        y_pred = model.predict(X_test)
        get_pr_auc_score(y_test, y_pred)