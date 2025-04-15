"""
applicaton of natural language processing:
chatbot
speech recogntion
topic identification
sentiment analysis
machine translation
"""

from nltk.tokenize import word_tokenize, sent_tokenize

#document: scene_one
scene_one = '/word.doc'
into_sentence = word_tokenize(scene_one)
tokenize_4th_sentence = word_tokenize(into_sentence[3])
unique_tokens = set(word_tokenize(scene_one))

import re
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.match('coconuts')
# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

match_digits_and_words = '(\w+|\d+)'
re.findall(match_digits_and_words, "he has 11 cats.")

ABCDza = [A-Za-z]+
digits = [0-9]

from ntlk.tokenize import TweetTokenizer, regexp_tokenize
pattern2 = r'#\w+'
hashtags = TweetTokenizer(regexp_tokenize[0], pattern2)

#Bag of words
from collections import Counter
counter = Counter(words)
counter.most_common(2)
# bag of words - simple way to convert words to numerical representations

#preprocessing
"""
tokenization
lowercasing words
lemmatization/stemming -- shortening words to their root stems
removing stop words, puncuations or unwanted tokens
"""
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

words = """The cat is in the box. the cat like to... the cat is now """

tokens = [w for w in word_tokenize(words.lower() if w.isalpha())]

remove_stops = [t for t in tokens if t not in stopwords.words('english')]

Counter(remove_stops).most_common(2)

#lemmatize
lemmatize = WordNetLemmatizer()
no_stopwords.lemmatize()

# classification model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

x = df['xxxx']
y = df[yyyy]

x_train, x_test....

count_model = CountVectorizer(stop_words='english')
count_model.fit_transform(x_train.values)
count_model.transform(x_test.values)

#using naive bayes for classification problem as it works with probability
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb_classifer = MultinomialNB()
metrics.accuracy_score(pred, y)

"""whole succint code for preprocessing"""

from collection import Counter
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocessing(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]

    lem = WordNetLemmatizer()
    words = [lem.lemmatize(word) for word in words]

    stop_words = set(stopwords.word('english'))
    words = [word for word in words if word not in stop_words]

    count = Counter(words)
    count = count.most_common()
    return count

# Example usage
text_example = "Natural Language Processing is fascinating and involves many interesting techniques."

preprocessed_text = preprocess_text(text_example)
print("Original Text:", text_example)
print("Preprocessed Text:", preprocessed_text) 

#part of speech pos_tagging
from nltk.tag import pos_tag
words = "Natural Language Processing is fascinating and involves many interesting techniques."

words = word_tokenize(sentence)
words = pos_tagging(words)

#name entity recognition NER chunk using nlt or spacer
from nltk import import ne_chunk

words = word_tokenize(sentence)
words = pos_tagging(words)
print(ne_chunk(words))