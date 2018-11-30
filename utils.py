import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

def plot_confusion_matrix(prediction, y_test):
    plt.figure(figsize=(15, 6))
    c_matrix = confusion_matrix(y_test, prediction)
    c_matrix_ = np.round(100 * c_matrix / c_matrix.sum(axis=1).reshape(-1, 1))
    plt.subplot(121)
    sns.heatmap(c_matrix, annot=True, fmt="d");
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplot(122)
    sns.heatmap(np.asarray(c_matrix_, dtype=int), annot=True, fmt="d")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def get_roc_auc_score(prediction, y_test):
    value = np.round(roc_auc_score(y_test, prediction[:,1]),4)
    print('ROC AUC score:',  value)
    
    return value

def get_pr_auc_score(prediction, y_test):
    value = np.round(average_precision_score(y_test, prediction[:, 1]),4)
    print('PR AUC score:',  value)
    
    return value
    
def plot_curves(prediction, y_test):
    tpr, fpr, _ = roc_curve(y_test, prediction[:,1])
    roc_auc = roc_auc_score(y_test, prediction[:,1])
    
    precision, recall, _ = precision_recall_curve(y_test, prediction[:,1])
    average_precision = average_precision_score(y_test, prediction[:,1])

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.step(tpr, fpr, color='b', alpha=0.2, where='post')
    plt.fill_between(tpr, fpr, step='post', alpha=0.2, color='b')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class ROC curve: ROC AUC={0:0.2f}'.format(roc_auc))

    plt.subplot(122)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    
def get_threshold_by_f1(prediction_proba, y_test, plot=False):
    thresholds = np.linspace(prediction_proba[:, 1].min(), prediction_proba[:, 1].max(), 300)
    curve = []
    for val in thresholds:
       
        prediction = np.asarray((prediction_proba[:, 1] >= val), int)
        curve.append(f1_score(prediction, y_test))

    argmax = np.argmax(curve)

    if plot == True:
        plt.figure(figsize=(14, 4))
        plt.plot(thresholds, curve)
        plt.plot([thresholds[argmax]] * 2, [np.min(curve), np.max(curve)])
    
    return thresholds[argmax]

def plot_probas(prediction_proba, y_test):
    plt.figure(figsize=(14, 4))
    probas = prediction_proba[:, 1]
    ax = sns.distplot(probas)
    proba_mean = probas.mean()
    proba_f1_best = get_threshold_by_f1(prediction_proba, y_test)
    plt.plot([proba_mean] * 2, ax.get_ylim(), label='Mean of probabilites')
    plt.plot([proba_f1_best] * 2, ax.get_ylim(), label='Best threshold by f1 score')
    plt.legend(fontsize=16)

def get_indices(data):
    current_ID = ''
    current_subset = []
    indices = []

    for index, ID_name in enumerate(data.ID):
        if ID_name == current_ID:
            current_subset.append(index)
        else:
            if index > 0: 
                indices.append(current_subset) # finish the subset and start new one
            current_ID = ID_name
            current_subset = [index]

    indices.append(current_subset) # for the last one

    assert len(indices) == len(data.ID.unique())
    return indices

def construct_sentences(data, column_name, dropna=True):
    indices = get_indices(data)
    sentences = []
    
    series = data[column_name]
    
    if not dropna:
        series = series.fillna('NAN' + column_name)

    ids = data.ID.unique()
    for index, ID_name in tqdm(enumerate(ids), total=len(ids)):
        if dropna:
            sentences.append(list(series.iloc[indices[index]].dropna()))
        else:
            sentences.append(list(series.iloc[indices[index]]))
    
    sentences = [' '.join(sent) for sent in sentences]
    
    return sentences

def prepare_features(DATA):
    indices = get_indices(DATA)
    
    vals = DATA.BETRAG / DATA.ANZAHL
    DATA.BETRAG = np.log(vals - np.min(vals, 0) +  1)
    
    betrag_mean = DATA.groupby(['ID'])['BETRAG'].mean()
    betrag_std = DATA.groupby(['ID'])['BETRAG'].std().fillna(0)
    betrag_min = DATA.groupby(['ID'])['BETRAG'].min()
    betrag_max = DATA.groupby(['ID'])['BETRAG'].max()
    betrag_median = DATA.groupby(['ID'])['BETRAG'].median()

    betrag_all = pd.concat([betrag_mean, betrag_std, betrag_min, betrag_max, betrag_median], axis=1, keys=
              ['BETRAG_mean', 'BETRAG_std', 'BETRAG_min', 'BETRAG_max', 'BETRAG_median']).reset_index()
    
    betrag_data = []
    ticks = np.linspace(2, 6, 100)
    ids = DATA.ID.unique()
    for i, Id in enumerate(ids):
        indices[i]
        betrag_data.append(np.histogram(DATA.iloc[indices[i]]['BETRAG'], ticks)[0])
        
    betrag_data = pd.DataFrame(betrag_data, columns = ['betrag' + str(np.round(val, 2)) for val in ticks[:-1]])
    betrag_data = pd.concat([pd.Series(ids, name='ID'), betrag_data], axis=1)
    
    faktor_mean = DATA.groupby(['ID'])['FAKTOR'].mean()
    faktor_std = DATA.groupby(['ID'])['FAKTOR'].std().fillna(0)
    faktor_min = DATA.groupby(['ID'])['FAKTOR'].min()
    faktor_max = DATA.groupby(['ID'])['FAKTOR'].max()
    faktor_median = DATA.groupby(['ID'])['FAKTOR'].median()

    faktor_all = pd.concat([faktor_mean, faktor_std, faktor_min, faktor_max, faktor_median], axis=1, keys=
              ['FAKTOR_mean', 'FAKTOR_std', 'FAKTOR_min', 'FAKTOR_max', 'FAKTOR_median']).reset_index()

    DATA.TYP.fillna(-1, inplace=True)
    typ_mean = DATA.groupby(['ID'])['TYP'].mean()
    typ_std = DATA.groupby(['ID'])['TYP'].std().fillna(0)
    typ_min = DATA.groupby(['ID'])['TYP'].min()
    typ_max = DATA.groupby(['ID'])['TYP'].max()
    typ_median = DATA.groupby(['ID'])['TYP'].median()

    typ_all = pd.concat([typ_mean, typ_std, typ_min, typ_max, typ_median], axis=1, keys=
              ['TYP_mean', 'TYP_std', 'TYP_min', 'TYP_max', 'TYP_median']).reset_index()
    
    DATA.RECHNUNGSBETRAG = np.log(DATA.RECHNUNGSBETRAG - np.min(DATA.RECHNUNGSBETRAG, 0) +  1)
    
    data = DATA.drop_duplicates(subset=['ID'])[['ID', 'RECHNUNGSBETRAG', 'ALTER', 'GESCHLECHT', 'VERSICHERUNG', 'target']].reset_index(drop=True)

    data = data.merge(betrag_all, on='ID', how='inner')
    data = data.merge(betrag_data, on='ID', how='inner')
    data = data.merge(typ_all, on='ID', how='inner')
    data = data.merge(faktor_all, on='ID', how='inner')
    
    return data
   