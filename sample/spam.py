import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

app = Flask(__name__)
global Classifier
global Vectorizer

# load data

data = pandas.read_csv('/home/smsc/NLP/smsspamcollection/SMSSpamCollection', sep='\t', encoding='latin-1',  names=["label", "message"])
data.rename(columns = {'label': 'v1', 'message': 'v2'}, inplace = True)
train_data = data[:6000]
test_data = data[4400:]

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)

message = request.args.get('message', '')
error = ''
predict_proba = ''
predict = ''

vectorize_message = Vectorizer.transform([message])
predict = Classifier.predict(vectorize_message)[0]
predict_proba = Classifier.predict_proba(vectorize_message).tolist()





