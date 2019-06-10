import pandas as pd
import os


META_FEATURES = ['amount', 'age', 'sex', 'ins_type', 'speciality']

cols_mapping = {
    'ID': 'id',
    'KORREKTUR': 'adj',
    'RECHNUNGSBETRAG': 'amount',
    'ALTER': 'age',
    'GESCHLECHT': 'sex',
    'VERSICHERUNG': 'ins_type',
    'FACHRICHTUNG': 'speciality',  # why only 0/1 ?
    'NUMMER': 'treatment',
    'NUMMER_KAT': 'treatment_type',
    'TYP': 'billing_type',
    'ANZAHL': 'num_treatments',
    'FAKTOR': 'factor',
    'BETRAG': 'cost',
    'ART': 'cost_type',
    'LEISTUNG': 'ben_type'
}

data_dir = 'data'


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df1 = pd.read_csv(
        os.path.join(data_dir, 'arzta_daten_anonym1.csv'), sep=';')
    df2 = pd.read_csv(
        os.path.join(data_dir, 'arzta_daten_anonym2.csv'), sep=';')
    df3 = pd.read_csv(
        os.path.join(data_dir, 'arzta_daten_anonym3.csv'), sep=';')
    df4 = pd.read_csv(
        os.path.join(data_dir, 'arzta_daten_anonym4.csv'), sep=';')

    data = pd.concat([df1, df2, df3, df4])
    data = data.reset_index(drop=True).reset_index()
    data = data.rename(columns={'index' : 'order_id'})

    columns_comma = [
        'RECHNUNGSBETRAG', 'FAKTOR', 'BETRAG', 
        'ALTER', 'KORREKTUR']
    data[columns_comma] = data[columns_comma].apply(
        lambda x: x.str.replace(',', '.'))
    for column in columns_comma:
        data[column] = pd.to_numeric(data[column], downcast='float')

    data = data.rename(columns=cols_mapping)

    data['target'] = data['adj'].astype(bool).astype(int)
    data['treatment'] = data['treatment'].fillna(value='<UNK>')
    data['treatment_type'] = data['treatment_type'].fillna(value='<UNK>')
    data['billing_type'] = data['billing_type'].fillna(value='<UNK>')
    data['cost_type'] = data['cost_type'].fillna(value='<UNK>')
    data['ben_type'] = data['ben_type'].fillna(value='<UNK>')


    # Removing of outliers
    data = data[data['treatment'] != '<UNK>'] # removed unknown treatments
    data = data.drop(columns=['cost_type']) # removed due unknown treatments
    data = data[data['adj'] >= 0] # removed, where korrektur < 0
    data = data[data['cost'] >= 0] # removed, where cost < 0
    data = data.drop(columns=['speciality']) # removed due to duplication
    data = data.reset_index(drop=True)

    data.to_pickle(os.path.join(data_dir, 'data.pkl'))