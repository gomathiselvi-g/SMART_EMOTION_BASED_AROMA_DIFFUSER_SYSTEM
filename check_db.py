import sqlite3
conn = sqlite3.connect('emotion_history.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', [t['name'] for t in tables])

cursor.execute('PRAGMA table_info(timer_settings)')
cols = cursor.fetchall()
print('timer_settings columns:', [c['name'] for c in cols])

cursor.execute('PRAGMA table_info(video_recordings)')
cols = cursor.fetchall()
print('video_recordings columns:', [c['name'] for c in cols])

for table in ['users', 'timer_settings', 'video_recordings']:
    cursor.execute('SELECT COUNT(*) as cnt FROM ' + table)
    print(table + ' count:', cursor.fetchone()['cnt'])

# Check a sample user
cursor.execute('SELECT id, full_name, email, face_enrolled, monitoring_enabled FROM users LIMIT 3')
for row in cursor.fetchall():
    print('User:', dict(row))

conn.close()
print('DB check complete.')
