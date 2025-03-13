import sqlite3

def init_db():
    conn = sqlite3.connect('ampe_bci.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            movement TEXT NOT NULL,
            score INTEGER NOT NULL,
            signal TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(file_name, movement, score, signal):
    conn = sqlite3.connect('ampe_bci.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions (file_name, movement, score, signal)
        VALUES (?, ?, ?, ?)
    ''', (file_name, movement, score, str(signal)))

    conn.commit()
    conn.close()
