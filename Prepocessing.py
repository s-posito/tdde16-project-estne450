import re
import unicodedata
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def process_queries(queries, vocab):
    queries = process_texts(queries)
    # Removal of words from queries that are not in the vocabulary
    res = []
    for query in queries:
        words = query.split()
        query = ""
        for word in words:
            if word in vocab:
                query += word + " "
        res.append(query)
    return res


def remove_accents(input_str):
    norm_txt = unicodedata.normalize('NFD', input_str)
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)


def process_texts(texts):
    res = []
    for text in texts:
        # Replace diacritics
        text = remove_accents(text)
        # Lowercase the document
        text = text.lower()
        # Remove punctuations
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        # Remove stop-words
        text = remove_stop_words(text)
        # Lemmatization of the text
        text = lemmatize(text)
        res.append(text)
    return res


def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = ""
    for word in word_tokens:
        if word not in stop_words:
            filtered_text += word + " "
    return filtered_text


def lemmatize(text):
    word_tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokens])
    return text
