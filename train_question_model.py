import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("trainprj.csv")

# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['question'])
y = data['type']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open("question_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Model training complete!")
