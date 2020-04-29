import pandas as pd
import datetime, nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_product(df_initial):
    print('BEGIN')
    temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
    temp = temp.reset_index(drop=False)
    # *****
    temp = df_initial.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
    nb_products_per_basket = temp.rename(columns={'InvoiceDate': 'Number of products'})
    nb_products_per_basket[:10].sort_values('CustomerID')
    # 2.2.1 Cancelling orders
    nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x: int('C' in x))
    df_check = df_initial[df_initial['Quantity'] < 0][['CustomerID', 'Quantity',
                                                       'StockCode', 'Description', 'UnitPrice']]
    df_check = df_initial[(df_initial['Quantity'] < 0) & (df_initial['Description'] != 'Discount')][
        ['CustomerID', 'Quantity', 'StockCode',
         'Description', 'UnitPrice']]
    df_cleaned = df_initial.copy(deep=True)
    df_cleaned['QuantityCanceled'] = 0

    entry_to_remove = []
    doubtfull_entry = []

    for index, col in df_initial.iterrows():
        if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue
        df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                             (df_initial['StockCode'] == col['StockCode']) &
                             (df_initial['InvoiceDate'] < col['InvoiceDate']) &
                             (df_initial['Quantity'] > 0)].copy()
        # _________________________________
        # Cancelation WITHOUT counterpart
        if (df_test.shape[0] == 0):
            doubtfull_entry.append(index)
        # ________________________________
        # Cancelation WITH a counterpart
        elif (df_test.shape[0] == 1):
            index_order = df_test.index[0]
            df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index)
            # ______________________________________________________________
        # Various counterparts exist in orders: we delete the last one
        elif (df_test.shape[0] > 1):
            df_test.sort_index(axis=0, ascending=False, inplace=True)
            for ind, val in df_test.iterrows():
                if val['Quantity'] < -col['Quantity']: continue
                df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
                entry_to_remove.append(index)
                break
    df_cleaned.drop(entry_to_remove, axis=0, inplace=True)
    df_cleaned.drop(doubtfull_entry, axis=0, inplace=True)
    remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
    # 2.2.2 StockCode
    list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)][
        'StockCode'].unique()
    # 2.2.3 Basket Price
    df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})

    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
    basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
    basket_price = basket_price[basket_price['Basket Price'] > 0]
    price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
    count_price = []
    for i, price in enumerate(price_range):
        if i == 0: continue
        val = basket_price[(basket_price['Basket Price'] < price) &
                           (basket_price['Basket Price'] > price_range[i - 1])]['Basket Price'].count()
        count_price.append(val)
    # 3. Insight on product categories
    # 3.1 Products Description
    is_noun = lambda pos: pos[:2] == 'NN'
    nltk.download('all')

    def keywords_inventory(dataframe, colonne='Description'):
        stemmer = nltk.stem.SnowballStemmer("english")
        keywords_roots = dict()  # collect the words / root
        keywords_select = dict()  # association: root <-> keyword
        category_keys = []
        count_keywords = dict()
        icount = 0
        for s in dataframe[colonne]:
            if pd.isnull(s): continue
            lines = s.lower()
            tokenized = nltk.word_tokenize(lines)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

            for t in nouns:
                t = t.lower()
                racine = stemmer.stem(t)
                if racine in keywords_roots:
                    keywords_roots[racine].add(t)
                    count_keywords[racine] += 1
                else:
                    keywords_roots[racine] = {t}
                    count_keywords[racine] = 1

        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:
                min_length = 1000
                for k in keywords_roots[s]:
                    if len(k) < min_length:
                        clef = k
                        min_length = len(k)
                category_keys.append(clef)
                keywords_select[s] = clef
            else:
                category_keys.append(list(keywords_roots[s])[0])
                keywords_select[s] = list(keywords_roots[s])[0]
        return category_keys, keywords_roots, keywords_select, count_keywords

    df_produits = pd.DataFrame(df_initial['Description'].unique()).rename(columns={0: 'Description'})
    keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)
    list_products = []
    print('MEDDLE')
    for k, v in count_keywords.items():
        list_products.append([keywords_select[k], v])
    list_products.sort(key=lambda x: x[1], reverse=True)
    # liste = sorted(list_products, key=lambda x: x[1], reverse=True)
    # 3.2 Defining product categories
    list_products = []
    for k, v in count_keywords.items():
        word = keywords_select[k]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
        if len(word) < 3 or v < 13: continue
        if ('+' in word) or ('/' in word): continue
        list_products.append([word, v])
    # ______________________________________________________
    list_products.sort(key=lambda x: x[1], reverse=True)
    # 3.2.1 Data encoding
    liste_produits = df_cleaned['Description'].unique()
    X = pd.DataFrame()
    for key, occurence in list_products:
        X.loc[:, key] = list(map(lambda x: int(key.upper() in x), liste_produits))
    threshold = [0, 1, 2, 3, 5, 10]
    label_col = []
    for i in range(len(threshold)):
        if i == len(threshold) - 1:
            col = '.>{}'.format(threshold[i])
        else:
            col = '{}<.<{}'.format(threshold[i], threshold[i + 1])
        label_col.append(col)
        X.loc[:, col] = 0

    for i, prod in enumerate(liste_produits):
        prix = df_cleaned[df_cleaned['Description'] == prod]['UnitPrice'].mean()
        j = 0
        while prix > threshold[j]:
            j += 1
            if j == len(threshold): break
        X.loc[i, label_col[j - 1]] = 1
    # 3.2.2 Creating clusters of products
    matrix = X.values
    for n_clusters in range(3, 10):
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    n_clusters = 5
    silhouette_avg = -1
    while silhouette_avg < 0.145:
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)

        # km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
        # clusters = km.fit_predict(matrix)
        # silhouette_avg = silhouette_score(matrix, clusters)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    # 3.2.3 Characterizing the content of clusters
    sample_silhouette_values = silhouette_samples(matrix, clusters)
    liste = pd.DataFrame(liste_produits)
    liste_words = [word for (word, occurence) in list_products]

    occurence = [dict() for _ in range(n_clusters)]

    for i in range(n_clusters):
        liste_cluster = liste.loc[clusters == i]
        for word in liste_words:
            if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
            occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))
    # Principal Component Analysis
    # pca = PCA()
    # pca.fit(matrix)
    # pca_samples = pca.transform(matrix)
    # pca = PCA(n_components=30)
    # matrix_9D = pca.fit_transform(matrix)
    # mat = pd.DataFrame(matrix_9D)
    # mat['cluster'] = pd.Series(clusters)
    # 4. Customer categories
    corresp = dict()
    for key, val in zip(liste_produits, clusters):
        corresp[key] = val
        # __________________________________________________________________________
    df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)
    # 4.1.1 Grouping products
    for i in range(5):
        col = 'categ_{}'.format(i)
        df_temp = df_cleaned[df_cleaned['categ_product'] == i]
        price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
        price_temp = price_temp.apply(lambda x: x if x > 0 else 0)
        df_cleaned.loc[:, col] = price_temp
        df_cleaned[col].fillna(0, inplace=True)
    # ___________________________________________
    # somme des achats / utilisateur & commande
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})
    # ____________________________________________________________
    # pourcentage du prix de la commande / categorie de produit
    for i in range(5):
        col = 'categ_{}'.format(i)
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
        basket_price.loc[:, col] = temp

    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
    basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])

    basket_price = basket_price[basket_price['Basket Price'] > 0]
    set_entrainement = basket_price[
        basket_price['InvoiceDate'] < pd.to_datetime('20111101', format='%Y%m%d', errors='ignore')]
    set_test = basket_price[basket_price['InvoiceDate'] >= pd.to_datetime('20111101', format='%Y%m%d', errors='ignore')]
    basket_price = set_entrainement.copy(deep=True)
    # 4.1.3 Consumer Order Combinations
    transactions_per_user = basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(
        ['count', 'min', 'max', 'mean', 'sum'])

    for i in range(5):
        col = 'categ_{}'.format(i)
        transactions_per_user.loc[:, col] = basket_price.groupby(by=['CustomerID'])[col].sum() / transactions_per_user[
            'sum'] * 100

    transactions_per_user.reset_index(drop=False, inplace=True)
    basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
    last_date = basket_price['InvoiceDate'].max().date()

    first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
    last_purchase = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

    test = first_registration.applymap(lambda x: (last_date - x.date()).days)
    test2 = last_purchase.applymap(lambda x: (last_date - x.date()).days)

    transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop=False)['InvoiceDate']
    transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop=False)['InvoiceDate']
    list_cols = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
    # _____________________________________________________________
    selected_customers = transactions_per_user.copy(deep=True)
    matrix = selected_customers[list_cols].values
    scaler = StandardScaler()
    scaler.fit(matrix)
    # print('Giá trị trung bình của các biến : \n' + 90 * '-' + '\n', scaler.mean_)
    scaled_matrix = scaler.transform(matrix)
    pca = PCA()
    pca.fit(scaled_matrix)
    pca_samples = pca.transform(scaled_matrix)
    n_clusters = 11
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100)
    kmeans.fit(scaled_matrix)
    clusters_clients = kmeans.predict(scaled_matrix)
    silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
    print('silhouette score: {:<.3f}'.format(silhouette_avg))
    pca = PCA(n_components=6)
    matrix_3D = pca.fit_transform(scaled_matrix)
    mat = pd.DataFrame(matrix_3D)
    mat['cluster'] = pd.Series(clusters_clients)
    selected_customers.loc[:, 'cluster'] = clusters_clients
    merged_df = pd.DataFrame()
    for i in range(n_clusters):
        test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
        test = test.T.set_index('cluster', drop=True)
        test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
        merged_df = pd.concat([merged_df, test])
    # _____________________________________________________
    merged_df.drop('CustomerID', axis=1, inplace=True)
    print('number of customers:', merged_df['size'].sum())

    merged_df = merged_df.sort_values('sum')
    print(merged_df.shape)
    # liste_index = []
    # for i in range(5):
    #     column = 'categ_{}'.format(i)
    #     liste_index.append(merged_df[merged_df[column] > 45].index.values[0])
    # # ___________________________________
    # liste_index_reordered = liste_index
    # liste_index_reordered += [s for s in merged_df.index if s not in liste_index]
    # # ___________________________________________________________
    # merged_df = merged_df.reindex(index=liste_index_reordered)
    # merged_df = merged_df.reset_index(drop=False)
    # columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
    # X = selected_customers[columns]
    # Y = selected_customers['cluster']
    return selected_customers
