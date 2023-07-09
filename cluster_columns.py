import pandas as pd
#import cluster as cl
from sqlalchemy import create_engine
from itertools import product
import spacy
from sklearn.cluster import KMeans
import numpy as np

def preprocessing(df):
    # simple random sampling
    mask = df.isnull().any(axis=1)
    filtered_df = df[~mask]
    replace_it = True if (len(filtered_df) < 100) else False
    simple_random_sample = filtered_df.sample(n=3, random_state=190, replace= replace_it)
    df = simple_random_sample

    string_values = []

    #combining values and appending together into a single array
    for column in df.columns:
        column_values = df[column].astype(str)
        joined_values = '$@'.join(column_values)
        string_values.append(joined_values)

    print(string_values)

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
            print(average_vector.shape)
            embeddings.append(average_vector)
    print(len(embeddings))
    return np.array(embeddings)

nlp = spacy.load('en_core_web_md')
# connecting to database
rowcount = 100
conn = create_engine('postgresql://postgres:password@localhost:5434/database1')
#query for first dataset
query1 = "SELECT * FROM public.order_reviews LIMIT %s" % rowcount
#query for second dataset
query2 = "SELECT * FROM public.orders LIMIT %s" % rowcount

df1 = pd.read_sql_query(query1.replace('%', '%%'), con = conn)
df2 = pd.read_sql_query(query2.replace('%', '%%'), con = conn)

data1 = preprocessing(df1)
data2 = preprocessing(df2)

data = np.concatenate((data1, data2), 0)
print(data1.shape, data2.shape, data.shape)

#lets do clustering
#kmeans clustering
def kmeans_clustering(features):
    kmeans = KMeans(init = "random", n_clusters= 5, n_init=10, max_iter= 100, random_state=53)
    kmeans.fit(features)
    print(kmeans.labels_)

kmeans_clustering(data)
print(df1.columns.tolist())
print(df2.columns.tolist())