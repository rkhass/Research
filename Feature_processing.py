from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def get_indices(list_of_id):
    current_ID = ''
    current_subset = []
    indices = []

    for index, ID_name in enumerate(list_of_id):
        if ID_name == current_ID:
            current_subset.append(index)
        else:
            if index > 0: 
                indices.append(current_subset) # finish the subset and start new one
            current_ID = ID_name
            current_subset = [index]

    indices.append(current_subset) # for the last one

    return indices   

def sort_indices(arr):
    arr = [[i, int(i[3:])] for i in arr]
    return [i[0] for i in sorted(arr, key=lambda dt: dt[1])]

def construct_sentences(data, column_name, dropna=True):
    sentences = []

    series = data[column_name]
    uniques = sort_indices(data.ID.unique())
    indices = get_indices(data.ID)

    if not dropna:
        series = series.fillna('NAN' + column_name)

    for index, ID_name in tqdm(enumerate(uniques), total=len(uniques)):
        if dropna:
            sentences.append(list(series.iloc[indices[index]].dropna()))
        else:
            sentences.append(list(series.iloc[indices[index]]))

    sentences = [' '.join(sent) for sent in sentences]

    return sentences

def adaptive_boards(arr, N=2000):
    cur_boarder = 0
    boards = [0.]
    counter = 0
    counters = []
    for cur_val in arr:
        if counter < N:
            cur_boarder = cur_val
            counter += 1
        else:
            if cur_val == cur_boarder:
                counter += 1
            else:
                counters.append(counter)
                boards.append(cur_boarder)
                cur_boarder = cur_val
                counter = 1 

    counters.append(counter)
    boards.append(cur_boarder)
    
    map_array = []
    for i, num in enumerate(counters):
        map_array += [str(i)] * num
            
    return map_array

def save_features(train, test, name):
    prefix = '25_'
    train.to_pickle('../Features/Train/' + prefix + name + '.pkl')
    test.to_pickle('../Features/Test/' + prefix + name + '.pkl')
    
def fit_transform_tf_idf(train_sent, test_sent, name):
    model_tf_idf = TfidfVectorizer()
    model_tf_idf.fit(train_sent.values)
    X_train, X_test = model_tf_idf.transform(train_sent.values), model_tf_idf.transform(test_sent.values) 

    names = [name + '_' + str(i) for i in range(X_train.shape[1])]

    df_train = pd.SparseDataFrame(X_train, columns=names).fillna(0).to_dense()
    df_train = pd.concat([pd.Series(train_sent.index), df_train], axis=1)

    df_test = pd.SparseDataFrame(X_test, columns=names).fillna(0).to_dense()
    df_test = pd.concat([pd.Series(test_sent.index), df_test], axis=1)

    return df_train, df_test