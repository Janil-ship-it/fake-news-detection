import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

fake = pd.read_csv("data/raw/Fake.csv")
true = pd.read_csv("data/raw/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)

df["text"] = df["title"] + " " + df["text"]

df = df[df["text"].str.strip() != ""]

df["clean_text"] = df["text"].apply(clean_text)

output = "data/processed/news_clean.csv"

df.to_csv(output, index=False)

print("Dataset preprocessing completed.")
print(f"Rows: {len(df)}")
print(f"Saved to: {output}")
