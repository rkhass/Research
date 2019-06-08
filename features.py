from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
# from glove import Corpus, Glove
from multiprocessing import Pool


def train_val_test_split(df):
    """
    Split the dataset into train, validation and test parts

    Parameters
    ----------
    df : DataFrame
        Dataframe with target column
    """

    assert 'target' in df.columns, 'Assign target column to your dataset'
    assert 'id' in df.columns, 'Assign id column to your dataset'

    parts = []
    for train_idx, test_idx in GroupKFold(10).split(X=df, groups=df['id']):
        parts.append(test_idx)
        
    train = np.concatenate(parts[:8]) # 80% of data
    val = np.concatenate(parts[8:9]) # 10% of data
    test = np.concatenate(parts[9:10]) # 10% of data

    X_train, X_val, X_test = [df.iloc[part, :].sort_index() 
                              for part in [train, val, test]]

    return X_train, X_val, X_test


def number_of_visits(df):
    """
    Calculate the number of visits per user
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with id column

    Returns
    -------
    visits: Series
    """

    assert 'id' in df.columns, 'Assign id column to your dataset'

    visits = df.groupby('id')['id'].count()
    visits.rename('visits', inplace=True)

    return visits

def meta_features(df, columns):
    """
    Return meta features that don't change over time and
    permanently belong to each ID

    Parameters
    ----------
    df : DataFrame
    columns: list
        List of columns to extract as meta features

    Returns
    -------
    df_meta : DataFrame
    """

    columns += ['id']
    df_meta = df.drop_duplicates(subset=['id'])[columns].reset_index(drop=True)
    df_meta = df_meta.set_index('id')
    return df_meta


def statistics(df, on):
    """
    Calculate general statistics of series

    Parameters
    ----------
    df : DataFrame

    on : str
        The column name to calculate features

    Returns
    -------
    features: DataFrame
    """

    assert 'id' in df.columns, 'Assign id column to your dataset'
    assert on in df.columns, 'Input column is not exists in the dataset'
    assert np.issubdtype(df[on].dtype, np.number), 'The column is not numeric'

    groupby = df.groupby('id')[on]

    s = groupby.sum()
    log_sum = np.log(s - s.min() + 1)
    mean = groupby.mean()
    std = groupby.std().fillna(0)
    minimum = groupby.min()
    maximum = groupby.max()
    median = groupby.median()

    funcs = ['log_sum', 'mean', 'std', 'min', 'max', 'median']

    keys = [on + '_' + func for func in funcs]

    features = pd.concat(
        [log_sum, mean, std, minimum, maximum, median],
        keys=keys, axis=1
        )

    return features


def sentences(df, on):
    assert 'id' in df.columns, 'Assign id column to your dataset'
    assert on in df.columns, 'Input column is not exists in the dataset'
    assert df[on].dtype == np.object, 'The column is not object'

    df[on] = df[on] + ' ' # insert spaces
    features = df.groupby('id')[on].apply(lambda dt: dt.sum())

    return features


class GloveEmbedding():
    def __init__(self, no_components=100, learning_rate=0.05):
        self.no_components = no_components
        self.learning_rate = learning_rate


    def fit(self, series, window=10, epochs=100, verbose=True):
        """
        Fit the GloVe model 

        Parameters
        ----------
        series : Series
            Train Series of sentences to fit a model
        window : int, optional
            The length of the (symmetric) context window 
            used for cooccurrence.
        epochs : int, optional
            The number of epochs to fit a model
        verbose : bool
        """
        sentences = series.values
        tagged_data = [
            sentence.lower().split(' ')[:-1] for sentence in sentences]

        corpus = Corpus()
        corpus.fit(tagged_data, window=window)
        
        model = Glove(
            no_components=self.no_components,
            learning_rate=self.learning_rate)
        model.fit(
            corpus.matrix, 
            epochs=epochs,
            no_threads=Pool()._processes, 
            verbose=verbose)
        model.add_dictionary(corpus.dictionary)

        self._model = model


    def predict(self, series, col_name='col'):
        """
        Predict the embedded vectors

        Parameters
        ----------
        series : Series
            Test Series of sentences
        col_name : string
            The prefix of columns to be used

        Returns
        -------
        features : DataFrame
            DataFrame of Embedded vectors 
        """
        sentences = series.values
        vectors = np.zeros((len(sentences), self.no_components))

        for i, sentence in enumerate(sentences):
            tagged_sentence = sentence.lower().split(' ')[:-1]
            inferred_vectors = np.zeros(self.no_components)
            for word in tagged_sentence:
                if word in self._model.dictionary:
                    inferred_vectors += (
                        self._model.word_vectors[self._model.dictionary[word]]
                        )
                
            vectors[i] = inferred_vectors
            
        columns = [col_name + '_' + str(i) for i in range(self.no_components)]
        features = pd.DataFrame(
            data=vectors, 
            index=series.index, 
            columns=columns)

        return features 


def get_target(df, on='target'):
    """
    Return target column

    Parameters
    ----------
    df : DataFrame
    on : string
        The name of target column

    Returns
    -------
    target : Series
    """
    assert on in df.columns, 'Assign target column to your dataset'
    
    target = df.groupby('id')[on].sum().astype(bool).astype(int)
    return target


def transform_tf_idf(train, val=None, test=None, col_name='col'):
    """
    Make a TF-IDF transformation for train and test sentences

    Parameters
    ----------
    train : DataFrame
    val :DataFrame
    test : DataFrame
    col_name : string

    Returns
    -------
    frames : tuple of DataFrames
    """

    model = TfidfVectorizer()
    X_train = model.fit_transform(train)

    names = [col_name + '_' + str(i) for i in range(X_train.shape[1])]

    df_train = pd.SparseDataFrame(
        data=X_train, 
        index=train.index, 
        columns=names).fillna(0).to_dense()

    if val is not None:
        X_val = model.transform(val)  
        df_val = pd.SparseDataFrame(
            data=X_val, 
            index=val.index, 
            columns=names).fillna(0).to_dense()

    if test is not None:
        X_test = model.transform(test)  
        df_test = pd.SparseDataFrame(
            data=X_test, 
            index=test.index, 
            columns=names).fillna(0).to_dense()

    frames = (
        df_train,
        df_val if val is not None else None, 
        df_test if test is not None else None)

    return frames


class Categorizer:
    def __init__(self, max_number=5000):
        self.max_number = max_number

    def _get_sorted_array(self, data):
        X = data[['id', self._on]]
        X = X.sort_values(self._on)
        array = X[self._on].values

        return array

    def _set_intervals(self, boundaries):
        intervals = []
        
        for i in range(len(boundaries) - 1):
            intervals.append([boundaries[i], boundaries[i+1]])
        
        self._intervals = intervals
        self._boundaries = boundaries

    def fit(self, data, on):
        self._on = on

        array = self._get_sorted_array(data)
        clusters, boundaries = cluster_sorted_values(array, max_number=self.max_number)
        self._set_intervals(boundaries)

    def predict(self, data):
        array = data[self._on].values

        clusters = np.zeros_like(array, int) - 1

        for cluster_id, interval in enumerate(self._intervals):
            idx = np.where((interval[0] < array) & (array <= interval[1]))[0]
            clusters[idx] = cluster_id

        outliers = np.where(clusters == -1)[0]
        if len(outliers) > 0:
            clusters[outliers] = cluster_id
            print('{0} outliers in data'.format(len(outliers)))

        clusters = ['cat_' + str(cluster) for cluster in clusters]

        return clusters


def cluster_sorted_values(array, max_number):
    """
    Cluster sorted numerical values into groups

    Parameters
    ----------
    array : ndarray
    max_number : int
        The maximum number of examples in a cluster.
        The size of a cluster can be changed if there are duplicate values.

    Returns
    -------
    clusters : list
        List of defined clusters for each value of array
    """

    head_value = 0
    cluster_size = 0
    cluster_sizes = []
    cluster_heads = [-1e-3]

    for value in array:
        if cluster_size < max_number:
            head_value = value
            cluster_size += 1
        else:
            if value == head_value:
                cluster_size += 1
            else:
                cluster_sizes.append(cluster_size)
                cluster_heads.append(head_value)
                head_value = value
                cluster_size = 1 

    cluster_sizes.append(cluster_size)
    cluster_heads.append(head_value)
    
    clusters = []
    for i, cluster_size in enumerate(cluster_sizes):
        clusters += [str(i)] * cluster_size
            
    return clusters, cluster_heads


def get_list_of_features():
    pass