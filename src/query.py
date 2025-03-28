import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# cursor.execute("DELETE FROM users WHERE name = 'Arslaan Siddiqui'")
cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM sqlite_sequence")

conn.commit()
conn.close()
print("All records deleted!")