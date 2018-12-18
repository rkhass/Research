import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_raw_data_to_one_file():
    data1 = pd.read_csv('../Data_original/arzta_daten_anonym1.csv', sep=';')
    data2 = pd.read_csv('../Data_original/arzta_daten_anonym2.csv', sep=';')
    data3 = pd.read_csv('../Data_original/arzta_daten_anonym3.csv', sep=';')
    data4 = pd.read_csv('../Data_original/arzta_daten_anonym4.csv', sep=';')
    DATA = pd.concat([data1, data2, data3, data4], axis=0)
    
    DATA.reset_index(drop=True, inplace=True)
    DATA.index.rename('Time', inplace=True)
    DATA.reset_index(inplace=True)
    
    columns_comma = ['RECHNUNGSBETRAG', 'FAKTOR', 'BETRAG', 'ALTER', 'KORREKTUR'] 
    DATA[columns_comma] = DATA[columns_comma].apply(lambda x: x.str.replace(',', '.'))
    for column in columns_comma:
        DATA[column] = pd.to_numeric(DATA[column], downcast='float')
        
    target = DATA.groupby(['ID'])['KORREKTUR'].apply(lambda dt: int(np.sign(dt).values[0])).to_frame(name='target')
    DATA = DATA.merge(target, on='ID', how='inner')
         
    return DATA

def separate_data(data, target, test_ID_size=0.1, test_Time_size=0.2):
    """Separates data into train and test parts
        
        Parameters
        ----------
        data : 
        
        target :
        
        test_ID_size :
        
        test_Time_size :
        
                
        Returns
        -------
        
        Train :
        
        Test :
        """
    
    X_train, X_test, y_train, y_test = train_test_split(target.ID, 
                                                        target.target, 
                                                        stratify  = target.target,
                                                        test_size = test_ID_size)

    train_ID = pd.concat([X_train, y_train], axis=1)
    test_ID  =  pd.concat([X_test,  y_test ], axis=1)

    train = data[data.ID.isin(train_ID.ID)]
    test  = data[data.ID.isin(test_ID.ID)]

    separator = int(data.Time.max() * (1 - test_Time_size))

    train = train[train.Time < separator]
    test = test[test.Time >= separator]

    av_record_length_train = train.ID.value_counts().mean()
    av_record_length_test  = test.ID.value_counts().mean()

    print('Average length of records in Train data: ', np.round(av_record_length_train, 2))
    print('Average length of records in Test  data: ', np.round(av_record_length_test, 2))

    percentage = (data.shape[0] - train.shape[0] - test.shape[0]) / data.shape[0]

    message = str(np.round(percentage * 100, 1)) + '% of data was dropped'
    print(message)

    return train, test
