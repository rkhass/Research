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

def get_pr_auc_score2(prediction, y_test):
    return average_precision_score(y_test, prediction)


def get_pr_auc_score(prediction, y_test):
    value = np.round(average_precision_score(y_test, prediction[:, 1]),4)
    print('PR AUC score:',  value)
    
    return value

def plot_pr_auc(stats):
    data_curr = []
    sizes = stats.visits_num.unique()
    sizes.sort()
    for i in sizes:
        tmp = stats[stats.visits_num==i]
        v = get_pr_auc_score2(tmp.prob.values, tmp.real.values)
        data_curr.append([i, v])
        
    data_df = pd.DataFrame(data_curr).dropna()
    data_df[2] = data_df[1].rolling(2).mean()
    plt.figure(figsize=(15, 6))
    plt.plot(*data_df[[0, 2]].values.T)    
    
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