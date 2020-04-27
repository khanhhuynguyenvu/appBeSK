import pandas as pd
from flask import jsonify
from models.Record import Record

df = pd.read_csv('uploads/data.csv', encoding="ISO-8859-1",
                 dtype={'CustomerID': str, 'InvoiceID': str}, nrows=10)
# df.to_csv(r'uploads/sample.csv', index=False, header=True)
x = Record(df.columns, df.iloc[0].values)
# print(x.toDictionary())
# print(df.shape[0])
print(x.toDictionary())
