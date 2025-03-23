from flask import Flask, render_template, request, redirect, session, url_for
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk
from db import Database

db = Database()

try:
    stop = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stop = stopwords.words('english')

try:
    lemmatizer = WordNetLemmatizer()
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
DATA_PATH = 'data/drugsComTrain.csv'

vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

rawtext = ""

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        raw_text = request.form['rawtext']

        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]
            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)
            return render_template('home.html', rawtext=raw_text, result=predicted_cond, top_drugs=top_drugs)
        else:
            raw_text = "There is no text to select"

    return render_template('home.html', rawtext=rawtext)

def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmitize_words)

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

@app.route('/index')
def index():
    condition = request.args.get('condition', '')
    if condition:
        reviews = db.search_by_condition(condition)
    else:
        reviews = db.fetch()
    return render_template('index.html', reviews=reviews)

@app.route('/add', methods=['GET', 'POST'])
def add_review():
    if request.method == 'POST':
        patientID = request.form['patientID']
        drugName = request.form['drugName']
        condition = request.form['condition']
        review = request.form['review']
        rating = request.form['rating']
        date = request.form['date']
        usefulCount = request.form['usefulCount']
        db.insert(patientID, drugName, condition, review, rating, date, usefulCount)
        return redirect(url_for('index'))
    return render_template('add.html')

@app.route('/edit/<string:patientID>', methods=['GET', 'POST'])
def edit_review(patientID):
    review = db.get_review(patientID)
    if request.method == 'POST':
        drugName = request.form['drugName']
        condition = request.form['condition']
        review_text = request.form['review']
        rating = request.form['rating']
        date = request.form['date']
        usefulCount = request.form['usefulCount']
        db.update(patientID, drugName, condition, review_text, rating, date, usefulCount)
        return redirect(url_for('index'))
    return render_template('edit.html', review=review)

@app.route('/delete/<string:patientID>')
def delete_review(patientID):
    db.remove(patientID)
    return redirect(url_for('index'))

@app.route('/home')
def home():
    return redirect(url_for('predict'))

heart_model_path = "heart.pkl"

# Load the data correctly
loaded_data = joblib.load(heart_model_path)

# Check if it's a tuple with two elements
if isinstance(loaded_data, (tuple, list)) and len(loaded_data) == 2:
    scaler, heart_model = loaded_data
else:
    heart_model = loaded_data
    scaler = None  # If no scaler was saved, handle it

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        data = [float(x) for x in request.form.values()]
        features = np.array(data).reshape(1, -1)

        if scaler:  # Scale features only if a scaler is available
            features = scaler.transform(features)

        prediction = heart_model.predict(features)
        output = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

        return render_template("heart.html", prediction_text=output)

    return render_template("heart.html")





if __name__ == "__main__":
    app.run(debug=False)
