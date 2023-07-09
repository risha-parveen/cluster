import pandas as pd
#import cluster as cl
from sqlalchemy import create_engine
import psycopg2 as psy
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import os,re,time,unicodedata,sys
from itertools import product
import spacy
nlp = spacy.load('en_core_web_md')



# connecting to database
rowcount = 100
conn = psy.connect(database ="database1", user="postgres", password="password", host="localhost", port="5434")

query = "SELECT * FROM public.order_reviews LIMIT %s" % rowcount
df = pd.read_sql_query(query.replace('%', '%%'), con = conn)

# simple random sampling
mask = df.isnull().any(axis=1)
filtered_df = df[~mask]
replace_it = True if (len(filtered_df) < 100) else False
simple_random_sample = filtered_df.sample(n=3, random_state=190, replace= replace_it)
print(simple_random_sample["review_comment_title"])
df = simple_random_sample

string_values = []

for column in df.columns:
    column_values = df[column].astype(str)
    joined_values = '$@'.join(column_values)
    string_values.append(joined_values)

print(string_values)

embeddings = []
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
print(len(embeddings))

def ngrams(string, n=3):
    #convert unicode to ascii
    try:
        if isinstance(string, unicode): #Python 2
            string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore') 
    except Exception as e:
        if isinstance(string, str): #Python 3
            string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore') 

    string = str(string)
    string = string.lower()
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def CreateTfIdfVectorizerAll3Grams():
    chars = '123456789:;+-<=>?@[\]^_`abcdefghijklmnopqrstuvwxyz{|}~' #ascii_lowercase can also be used instead of chars, but only includes a-z
    allPossibleTrigrams = [''.join(i) for i in product(chars, repeat = 3)]
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    vectorizer.fit(allPossibleTrigrams)
    return vectorizer
