import pandas as pd
#import cluster as cl
from sqlalchemy import create_engine
from itertools import product
import spacy
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

sample_values = []
def preprocessing(df):
    # simple random sampling
    mask = df.isnull().any(axis=1)
    filtered_df = df[~mask]
    replace_it = True if (len(filtered_df) < 100) else False
    simple_random_sample = filtered_df.sample(n=100, random_state=190, replace= replace_it)
    df = simple_random_sample
    sample_values = df.iloc[0].values.tolist()
    string_values = []

    #combining values and appending together into a single array
    for column in df.columns:
        column_values = df[column].astype(str)
        joined_values = '$@'.join(column_values)
        string_values.append(joined_values)

    embeddings = []

    #finding the gloVe vectors
    for string in string_values:
        words = string.split('$@')
        vector_sum = 0
        count = 0
        
        for word in words:
            token = nlp(word)
            
            if token.has_vector:
                vector_sum += token.vector
                count += 1
        if count > 0:
            average_vector = vector_sum / count
            embeddings.append(average_vector)
        else:
            embeddings.append(np.zeros(shape=(300,)))
    return np.array(embeddings), sample_values

def add_to_table(labels, clusterLabel):
    dataframe_data[clusterLabel] = labels
    dataframe = pd.DataFrame(dataframe_data)
    return dataframe

def kmeans_clustering(features):
    kmeans = KMeans(init = "random", n_clusters= 5, n_init=10, max_iter= 1000, random_state=23)
    kmeans.fit(features)
    table = add_to_table(kmeans.labels_, 'kmeans')
    return table

def agglomerative_clustering(features):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(features)
    table = add_to_table(labels, 'agglo')
    return table

def dbscan_clustering(features):
    clustering = DBSCAN(eps=25, min_samples=2).fit(features)
    table = add_to_table(clustering.labels_, 'dbscan')
    return table

nlp = spacy.load('en_core_web_md')
# connecting to database
rowcount = 100
conn = create_engine('postgresql://postgres:password@localhost:5434/database1')
#query for first dataset
query1 = "SELECT * FROM public.order_items LIMIT %s" % rowcount
#query for second dataset
query2 = "SELECT * FROM public.order_payments LIMIT %s" % rowcount

df1 = pd.read_sql_query(query1.replace('%', '%%'), con = conn)
df2 = pd.read_sql_query(query2.replace('%', '%%'), con = conn)

data1, sample1 = preprocessing(df1)
data2, sample2 = preprocessing(df2)

data = np.concatenate((data1, data2), 0)

dataframe_data = {
    'tables': ['table 1'] + [' '] * (len(df1.columns.tolist()) - 1) +
      ['table 2'] + [' '] * (len(df2.columns.tolist()) - 1),
    'columns': df1.columns.tolist() + df2.columns.tolist(),
    'sample': sample1 + sample2
}

#lets do clustering
#kmeans clustering (partition-based)
kmeans_result = kmeans_clustering(data)

#agglomerative clustering (heirarchial-clustering)
agglo_result = agglomerative_clustering(data)

#dbscan (density based)
dbscan_result = dbscan_clustering(data)
print(dbscan_result)

