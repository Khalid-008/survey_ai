import pyodbc

conn = pyodbc.connect(
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    r"SERVER=KHALID;"
    r"DATABASE=VMS;"
    r"Trusted_Connection=yes;"
)