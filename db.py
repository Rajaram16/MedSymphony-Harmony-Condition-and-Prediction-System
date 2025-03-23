import sqlite3

class Database:
    def __init__(self, db="harmony_medications.db"):
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute('''
        CREATE TABLE IF NOT EXISTS drug_reviews (
            patientID TEXT PRIMARY KEY,
            drugName TEXT,
            condition TEXT,
            review TEXT,
            rating INTEGER,
            date TEXT,
            usefulCount INTEGER
        )''')
        self.con.commit()

    def insert(self, patientID, drugName, condition, review, rating, date, usefulCount):
        self.cur.execute("INSERT INTO drug_reviews (patientID, drugName, condition, review, rating, date, usefulCount) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (patientID, drugName, condition, review, rating, date, usefulCount))
        self.con.commit()

    def fetch(self):
        self.cur.execute("SELECT * FROM drug_reviews LIMIT 10")
        return self.cur.fetchall()

    def get_review(self, patientID):
        self.cur.execute("SELECT * FROM drug_reviews WHERE patientID=?", (patientID,))
        return self.cur.fetchone()

    def update(self, patientID, drugName, condition, review, rating, date, usefulCount):
        self.cur.execute("UPDATE drug_reviews SET drugName=?, condition=?, review=?, rating=?, date=?, usefulCount=? WHERE patientID=?",
                         (drugName, condition, review, rating, date, usefulCount, patientID))
        self.con.commit()

    def remove(self, patientID):
        self.cur.execute("DELETE FROM drug_reviews WHERE patientID=?", (patientID,))
        self.con.commit()

    def search_by_condition(self, condition):
        self.cur.execute("SELECT * FROM drug_reviews WHERE condition LIKE ?", ('%' + condition + '%',))
        return self.cur.fetchall()
