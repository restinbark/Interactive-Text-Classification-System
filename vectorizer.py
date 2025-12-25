from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text

def build_vectorizer(texts):
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X
