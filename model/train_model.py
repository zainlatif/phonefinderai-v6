import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data
df = pd.read_csv('../data/training_data.csv')
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# Features (X) and labels (y)
X = df['query']
y = df['label'].str.lower()

# Vectorize the text queries
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train classifier
clf = LogisticRegression()
clf.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(clf, '../model/model.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")