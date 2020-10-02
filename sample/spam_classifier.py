# importing the Dataset
import pandas as pd

messages = pd.read_csv('/home/smsc/NLP/smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])

# Data cleaning and preprocessing
import re
import nltk

# print(messages.head())
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import openpyxl
from pathlib import Path
import pandas as pd

# Setting the path to the xlsx file:
xlsx_file = Path('/home/smsc/Downloads', 'Spam_content.xlsx')
# loading the xlsx file
wb_obj = openpyxl.load_workbook(xlsx_file)
# xlsx worksheeet object
sheet = wb_obj.active
data1 = [i.value for i in sheet["A"]]

# data of csv
import openpyxl
from pathlib import Path
import pandas as pd

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# creating B-O-W modelprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# should take most frequent values, top 5000 values(1's and 0's)
cv = TfidfVectorizer()
# x independent feature (inputs data of 0's and 1's)
x = cv.fit_transform(corpus).toarray()
# taking spam out of spam/ham-->label-->so convert this into dummy variable, so that machine can understand

y = pd.get_dummies(messages['label'])
print(y)
# taking spam column values only
y = y.iloc[:, 1].values
# y dependent feature (1's and 0's of o/p results)
# Train test split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)
y_pred = spam_detect_model.predict(x_test)

# prediction with user text
clf = MultinomialNB()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
message = "You have won Apple iphone"
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)

from sklearn.multiclass import *
from sklearn.svm import *
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(corpus)
Classifier.fit(vectorize_text, y)

error = ''
predict_proba = ''
predict = ''

vectorize_message = Vectorizer.transform([message])
predict = Classifier.predict(vectorize_message)[0]
predict_proba = Classifier.predict_proba(vectorize_message).tolist()
print(predict_proba)
print(predict)


"""
abc = len(data1)
spam_len = 1
j = 0
while spam_len < abc:
    message = str(data1[spam_len])
    data = [str(message).lower()]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    if int(my_prediction[0]) == 1:
        pass
    else:
        j = j + 1
        print(message)

    spam_len = spam_len + 1
print(j)
"""
# confusion matrix 2*2 matrix for how many elements correctly predicted
from sklearn.metrics import confusion_matrix, accuracy_score

confusion_m = confusion_matrix(y_test, y_pred)
# for checking accuracy of predicted 2*2 matrix of the above

accuracy = accuracy_score(y_test, y_pred)


