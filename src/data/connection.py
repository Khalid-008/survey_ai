import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE"),
            connection_timeout=60,
            buffered=True
        )