import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
df = pd.read_csv('/home/smsc/NLP/emails.csv', names=["text", "spam"])
# Data cleaning and preprocessing
import re
import nltk
print(len(df))
# print(messages.head())
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []

for i in range(1, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = TfidfVectorizer()
# x independent feature (inputs data of 0's and 1's)
x = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(df['spam'])
y = y.iloc[:, 1].values
# Train test split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)
y_pred = spam_detect_model.predict(x_test)

# prediction with user text
clf = MultinomialNB()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
message = ""
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)
