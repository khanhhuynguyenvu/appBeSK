from functools import lru_cache
import pandas as pd


@lru_cache(None)
def get_records_cache(numberOfRow: int):  # Return data frame once
    print('Number of records: ', numberOfRow)
    df = pd.read_csv('uploads/Data.csv', encoding="ISO-8859-1",
                     dtype={'CustomerID': str, 'InvoiceID': str}, nrows=numberOfRow)
    return df


def get_process_df():
    return 1
