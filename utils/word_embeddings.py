from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

def word_embeddings(text):
    model = Word2Vec(sentences=[text], vector_size=100, window=5, min_count=1, workers=4)
    word_embeddings_weights = {word: model.wv[word].sum() for word in text}
    return word_embeddings_weights
