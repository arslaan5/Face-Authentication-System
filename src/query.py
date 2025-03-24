import sqlite3
import os


DB_PATH = os.path.abspath(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# cursor.execute('SELECT name, face_encoding FROM users')
# rows = cursor.fetchall()
cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM sqlite_sequence")

conn.commit()
conn.close()
print("All records deleted!")