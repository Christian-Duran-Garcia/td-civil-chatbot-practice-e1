import time
time.clock = time.perf_counter
from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlalchemy as sa
import numpy as np

### ---------- SQL Connection ---------- ###

# Create connection to database using sqlalchemy
connection_string = "mysql+mysqlconnector://root:@localhost:3306/chatbot_test"
engine = sa.create_engine(connection_string, echo=True)

# Start engine and select all data from faqs table into pandas df
with engine.begin() as conn:
    df = pd.read_sql_query(sa.text("select * from faqs"), conn)

### ---------- Data Manipulation ---------- ###

# Clean data
df.dropna(inplace=True)

# Fit the vectorizer
vect = TfidfVectorizer().fit(np.concatenate((df["Question"], df["Answer"])))

# Get the vectors for the questions specifically
questions_vect = vect.transform(df["Question"])

### ---------- FLASK APP ----------- ###

# Name the app
app = Flask(__name__)

# POST method for asking a question and getting an answer from chatbot
@app.post("/question")
def ask_question():
    request_data = request.get_json()
    vect_user_question = vect.transform([request_data["question"]])
    similarity = cosine_similarity(vect_user_question, questions_vect)
    closest_answer = np.argmax(similarity, axis=1)
    return_data = {"answer" : df["Answer"].iloc[closest_answer].values[0]}
    return return_data, 201