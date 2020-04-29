import pandas as pd

from functions.cluster.ClusterProduct import cluster_product
from models.RecordList import RecordList


def get_final_data(df_initial):
    df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])
    df_initial.dropna(axis=0, subset=['CustomerID'], inplace=True)
    df_initial.drop_duplicates(inplace=True)
    # return get_clean_data(df_initial)
    # return RecordList(get_clean_data(df_initial)).toList()
    return RecordList(cluster_product(df_initial)).toList()


def get_clean_data(df_initial):
    df_cleaned = df_initial.copy(deep=True)
    df_cleaned['QuantityCanceled'] = 0
    entry_to_remove = []
    doubtful_entry = []
    for index, col in df_initial.iterrows():
        if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue
        df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                             (df_initial['StockCode'] == col['StockCode']) &
                             (df_initial['InvoiceDate'] < col['InvoiceDate']) &
                             (df_initial['Quantity'] > 0)].copy()
        if df_test.shape[0] == 0:
            doubtful_entry.append(index)
        elif df_test.shape[0] == 1:
            index_order = df_test.index[0]
            df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index)
        elif df_test.shape[0] > 1:
            df_test.sort_index(axis=0, ascending=False, inplace=True)
            for ind, val in df_test.iterrows():
                if val['Quantity'] < -col['Quantity']:
                    continue
                df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
                entry_to_remove.append(index)
                break
    df_cleaned.drop(entry_to_remove, axis=0, inplace=True)
    df_cleaned.drop(doubtful_entry, axis=0, inplace=True)
    df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
    df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
    return df_cleaned
