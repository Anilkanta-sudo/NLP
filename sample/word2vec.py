import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph = """This study was a preliminary study of high school student value changes because of the terrorist 
attack on the U.S. The major limitations of this study were that the student population was from California and might 
not truly represent all high school students in the U.S. Further, this study could not be considered a truly 
longitudinal study because of privacy issues that prevented the researchers from identifying all the students who 
returned surveys before the attack. In addition, the senior class had graduated the previous year, and a much larger 
freshman class entered the school. These issues not only made the samples similar, but also different in their 
composition. The researchers will conduct periodic studies to explore whether these value changes are permanent and 
continue into adulthood. We do not know what if any changes will take place in their values as they grow older, 
and we will continue to explore their values in our longitudinal studies of the impact of the 9/11 terrorist attacks."""

# preprocessing the data
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# preaparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if not word in set(stopwords.words('english'))]

# training the word2vec model
model = Word2Vec(sentences, min_count=1)

words = model.wv.vocab
print(words)

# Finding word Vectors
vector = model.wv['issues']

# Test similar words
similar = model.wv.most_similar('terrorist')

