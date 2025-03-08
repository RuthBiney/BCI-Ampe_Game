import sqlite3

# ✅ Create Database
conn = sqlite3.connect("eeg_data.db")
cursor = conn.cursor()

# ✅ Create Tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS eeg_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    raw_signal BLOB,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    eeg_data_id INTEGER,
    predicted_movement TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(eeg_data_id) REFERENCES eeg_data(id)
)
""")

conn.commit()
conn.close()
print("✅ Database setup complete!")
