import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append('../')

# import numpy as np
import pandas as pd
from features import sentences


if __name__ == '__main__':
    data = pd.read_pickle('data/data.pkl')
    
    features = ['treatment', 'treatment_type', 'ben_type']

    for feature in features:
        print(feature)
        s = sentences(data, on=feature)
        s.to_pickle('data/' + feature + '.pkl')