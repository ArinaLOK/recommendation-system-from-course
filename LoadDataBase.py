import os
import psycopg2
import time
import pandas as pd
from database import postgres_connection

df = pd.read_sql("select * from public.user_data", postgres_connection())
print(df.head())
print(df.count)
df.to_csv("user_data.csv", index=False)

df = pd.read_sql("select * from public.post_text_df", postgres_connection())
print(df.head())
print(df.count)
df.to_csv("post_text_df.csv", index=False)

df = pd.read_sql("select * from public.feed_data LIMIT 10000000", postgres_connection())
print(df.head())
print(df.count)
df.to_csv("feed_data.csv", index=False)