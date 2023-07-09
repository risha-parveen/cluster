import pandas as pd
#import cluster as cl
from sqlalchemy import create_engine
import psycopg2 as psy

rowcount = 100
#conn = create_engine('postgresql://postgres:risha@localhost:5432/database1')
conn = psy.connect(database ="database1", user="postgres", password="password", host="localhost", port="5434")

query = "SELECT * FROM public.customers LIMIT %s" % rowcount
df = pd.read_sql_query(query.replace('%', '%%'), con = conn)
print(df)
