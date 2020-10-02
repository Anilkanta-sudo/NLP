import pandas as pd

df = pd.read_csv('/home/smsc/NLP/train.csv')
df.head()
# get the independent features
x = df.drop('label', axis=1)
x.head()
# get the dependent fetsures
y = df['label']
y.head()
df.shape

# data preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# removing Nan values
df = df.dropna()
messages = df.copy()
# adding indexes for Nan positions
messages.reset_index(inplace=True)
messages.head(10)
# print(messages.head(10)) print(messages['title'][6])
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Applying Countvectorizing
# creating Bag Of Words ngram with combination 1,2,3
cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
x = cv.fit_transform(corpus).toarray()
x.shape
y = messages['label']
# Train test split
# Divide the dataset into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

cv.get_feature_names()[:20]
cv.get_params()

count_df = pd.DataFrame(x_train, columns=cv.get_feature_names())
count_df.head()
import matplotlib.pyplot as plt

import itertools
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()


from sklearn import metrics
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
score


y_train.shape

#Passive Aggressive Classifier Algorithm

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter=50)

linear_clf.fit(x_train, y_train)
pred = linear_clf.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# Multinomial Classifier with Hyperparamete

classifier=MultinomialNB(alpha=0.1)

previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(x_train,y_train)
    y_pred=sub_classifier.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# Get Features names
feature_names = cv.get_feature_names()

classifier.coef_[0]

# Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]

# Most fake
sorted(zip(classifier.coef_[0], feature_names))[:5000]

# HashingVectorizer

hs_vectorizer = HashingVectorizer(n_features=5000,non_negative=True)
X = hs_vectorizer.fit_transform(corpus).toarray()
X.shape


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(plot_confusion_matrix(cm, classes=['FAKE', 'REAL']))


