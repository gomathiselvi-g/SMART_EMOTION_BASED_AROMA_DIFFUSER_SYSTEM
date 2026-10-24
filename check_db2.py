import sqlite3
conn = sqlite3.connect('emotion_history.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute('SELECT id, full_name, email, face_enrolled, monitoring_enabled FROM users')
for row in cursor.fetchall():
    print('User:', dict(row))
cursor.execute('SELECT * FROM timer_settings')
for row in cursor.fetchall():
    print('TimerSettings:', dict(row))
cursor.execute('SELECT id, user_id, file_name, started_at, ended_at, duration_seconds FROM video_recordings ORDER BY id DESC LIMIT 5')
for row in cursor.fetchall():
    print('Recording:', dict(row))
conn.close()
