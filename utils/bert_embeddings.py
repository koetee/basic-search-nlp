from bert_embedding import BertEmbedding
import numpy as np

def bert_embeddings(text, query):
    bert_embedding = BertEmbedding()
    text_embedding = np.mean(bert_embedding([text])[0][1], axis=0)
    query_embedding = np.mean(bert_embedding([query])[0][1], axis=0)

    similarity = np.dot(text_embedding, query_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(query_embedding))
    bert_weights = {word: similarity for word in text.split()}
    return bert_weights
