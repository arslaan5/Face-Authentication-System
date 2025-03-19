import sqlite3

conn = sqlite3.connect('face_recognition.db')
cursor = conn.cursor()

# cursor.execute('SELECT name, face_encoding FROM users')
# rows = cursor.fetchall()
cursor.execute('DELETE FROM users')
cursor.execute("DELETE FROM sqlite_sequence")

conn.commit()
conn.close()
print("All records deleted!")