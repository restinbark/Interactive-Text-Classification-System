
from sklearn.naive_bayes import MultinomialNB

def train_model(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model
