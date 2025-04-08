import os
import sqlite3


def create_database(conn):
    """Create the SQLite database and users table if they do not exist

    Args:
        conn (SQLite3 conn): Connection to the SQLite database
    """

    try:
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB NOT NULL,
            registration_date TEXT
        )
        ''')

        conn.commit()
        print("Database and table created successfully or already exists!")
    
    except sqlite3.OperationalError as e:
        print("Database error:", e)
    
    except sqlite3.Error as e:
        print("SQLite error:", e)

def find_project_root(current_path=None, marker="requirements.txt"):
    if current_path is None:
        current_path = os.getcwd()

    while True:
        if marker in os.listdir(current_path):
            return current_path
        new_path = os.path.dirname(current_path)
        if new_path == current_path:
            # Reached the filesystem root
            raise FileNotFoundError(f"Marker '{marker}' not found")
        current_path = new_path

root_path = find_project_root()
DB_PATH = os.path.join(root_path, "face_recognition.db")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    create_database(conn)
    conn.close()
