import os
import numpy as np
import pandas as pd
import sqlite3 as sql
from datetime import datetime

conn = sql.connect(r'backend\ArtificialIntelligence\data\soxx.db')
c = conn.cursor()

# Create the table
c.execute("""
    CREATE TABLE IF NOT EXISTS soxxData (
    High DOUBLE,
    Low DOUBLE,
    Volume DOUBLE,
    Open DOUBLE,
    Close DOUBLE,
    Return DOUBLE,
    RSI DOUBLE
    )   
""")
conn.commit()

def convert_data(path):
    data = pd.read_csv(path)
    return data

data = convert_data("backend/ArtificialIntelligence/soxxData.csv")

print(f"Values:\n{data}")
print(f"Shape of Values: {data.shape}")

data.to_sql('soxxData', conn, if_exists='append', index=False)

result = pd.read_sql_query("SELECT * FROM soxxData", conn)
print(f"Data in SQL Table:\n{result}")

conn.close()
