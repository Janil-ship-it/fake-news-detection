import joblib
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/processed/news_clean.csv")

print("Rows before cleaning:", len(df))

# Remove rows with missing text
df = df.dropna(subset=["clean_text"])

# Convert to string
df["clean_text"] = df["clean_text"].astype(str)

# Remove blank rows
df = df[df["clean_text"].str.strip() != ""]

print("Rows after cleaning:", len(df))

X = df["clean_text"]
y = df["label"]

# -------------------------
# Train/Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# -------------------------
# Bag of Words
# -------------------------

bow = CountVectorizer(max_features=5000)

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

joblib.dump(bow, "models/bow_vectorizer.pkl")
joblib.dump(X_train_bow, "models/X_train_bow.pkl")
joblib.dump(X_test_bow, "models/X_test_bow.pkl")

# -------------------------
# TF-IDF
# -------------------------

tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train_tfidf, "models/X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "models/X_test_tfidf.pkl")

# -------------------------
# Word2Vec
# -------------------------

sentences = [sentence.split() for sentence in X_train]

word2vec = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
)

def average_embedding(text):

    vectors = []

    for word in text.split():

        if word in word2vec.wv:
            vectors.append(word2vec.wv[word])

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)

X_train_embed = np.array(
    [average_embedding(text) for text in X_train]
)

X_test_embed = np.array(
    [average_embedding(text) for text in X_test]
)

joblib.dump(word2vec, "models/word2vec.model")
joblib.dump(X_train_embed, "models/X_train_embed.pkl")
joblib.dump(X_test_embed, "models/X_test_embed.pkl")

joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))