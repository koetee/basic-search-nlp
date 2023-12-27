import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymystem3 import Mystem
from nltk.stem import SnowballStemmer


class NLPModel:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('russian'))
        self.vectorizer = TfidfVectorizer()
        self.mystem = Mystem()

    def process_text(self, text):
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        return lemmatized_tokens

    def tokenize(self, text):
        words = word_tokenize(text, language='russian')
        return words

    def remove_stopwords(self, tokens):
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return filtered_tokens

    def lemmatize(self, tokens):
        stemmer = SnowballStemmer("russian")

        lemmatized_tokens = [stemmer.stem(word) for word in tokens]
        return lemmatized_tokens

    def calculate_similarity(self, text1, text2):
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return similarity[0][0]
