import os
import numpy as np
import yaml
from yaml.loader import SafeLoader
import sqlite3 as sql
from datetime import datetime


conn = sql.connect('users.db')

c = conn.cursor()

c.execute("""
    CREATE TABLE IF NOT EXISTS soxxData (
    High DOUBLE,
    Low DOUBE,
    Volume DOUBLE,
    Open DOUBLE,
    Close DOUBLE,
    Return DOUBLE,
    RSI DOUBLE
    )   
""")
conn.commit()

def add_daily_entry(row):
    c.execute("""
""")
