import sqlite3
from src.components.create_db import DB_PATH


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# cursor.execute("DELETE FROM users WHERE name = 'Arslaan Siddiqui'")
cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM sqlite_sequence")

conn.commit()
conn.close()
print("All records deleted!")