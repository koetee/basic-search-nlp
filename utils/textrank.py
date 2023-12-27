from summa import keywords

def textrank(text):
    text_rank_keywords = keywords.keywords(text)
    text_rank_weights = {word: 1.0 for word in text_rank_keywords.split()}
    return text_rank_weights
