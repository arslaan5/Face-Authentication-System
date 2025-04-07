import sqlite3
import os
from dotenv import load_dotenv
from src.components.utils import DB_PATH

load_dotenv()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# cursor.execute("DELETE FROM users WHERE name = 'Arslaan Siddiqui'")
cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM sqlite_sequence")

conn.commit()
conn.close()
print("All records deleted!")