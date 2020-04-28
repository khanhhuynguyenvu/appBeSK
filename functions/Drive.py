import pandas as pd
import requests
from io import StringIO
from functools import lru_cache


@lru_cache(None)
def get_Drive_csv(google_drive_url, numberOfRow=None):
    # orig_url = 'https://drive.google.com/open?id=1LGYpja_MiEJa2i_iNlICMjGkzSXqR-O8'
    fluff, file_id = google_drive_url.split('=')
    download_url = 'https://drive.google.com/uc?export=download&id=' + file_id
    url = requests.get(download_url).text
    csv_raw = StringIO(url)
    df = pd.read_csv(csv_raw, encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str}, nrows=numberOfRow)
    return df
