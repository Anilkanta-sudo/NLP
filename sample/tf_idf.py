import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

paragraph = """Policies of privatisation should be considered as responses to several distinct pressures. First, 
privatisation is a response by the state to internal forces such as increasing fiscal problems (O’Connor, 
1973). It provides a means of lessening the state’s fiscal responsibilities by encouraging the development of private 
alternatives which, theoretically at least, do not draw upon the state’s financial reserves. Second, the promotion of 
private sector activity is a response to pressures originating ‘outside’ the state apparatus. These include demands 
from people who see a large state bureaucracy as inefficient and wasteful, demands from business interests who claim 
that they can overcome these inefficiencies, and pressures from client groups who seek to reduce their dependency on 
the welfare state by having more control over the services on which they depend. Clearly, this variety of calls for 
privatisation means that it is not a process with a uniform outcome; there exists a correspondingly wide variety of 
forms of privatisation."""

wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# creating TF-IDF model
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()
print(x)