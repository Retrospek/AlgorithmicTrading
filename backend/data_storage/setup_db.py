import psycopg2

conn = psycopg2.connect(
    database="stockData",
    user="your_username",
    password="your_password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS stockData (
    date DATE,
    ticker VARCHAR(10),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume INTEGER
)
""")
conn.commit()