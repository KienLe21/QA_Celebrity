import sqlite3

# Kết nối với database SQLite
db_conn = sqlite3.connect("embeddings.db")
cursor = db_conn.cursor()

# Kiểm tra số lượng dòng trong database
cursor.execute("SELECT COUNT(*) FROM contexts")
row_count = cursor.fetchone()[0]
print(f"Total rows in database: {row_count}")

# Hiển thị 5 dòng đầu tiên để kiểm tra
cursor.execute("SELECT * FROM contexts LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Đóng kết nối
db_conn.close()