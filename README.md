# AI-Powered Fake News Detection Using Text Classification

## Overview

This project was developed as part of a **30-Day AI/ML Summer Internship**. It implements a complete Natural Language Processing (NLP) pipeline to classify news articles as **Fake** or **Real** using multiple machine learning algorithms. The project includes data preprocessing, feature engineering, model training, evaluation, and report generation.

---

## Objectives

* Detect fake news using text classification techniques.
* Compare the performance of multiple machine learning algorithms.
* Evaluate models using standard classification metrics.
* Generate reproducible results and visualizations.

---

## Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

Dataset files:

* `Fake.csv`
* `True.csv`

Each news article is assigned a binary label:

| Label | Meaning   |
| ----: | --------- |
|     0 | Fake News |
|     1 | Real News |

---

## Project Structure

```text
fake-news-detection
│
├── data
│   ├── raw
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed
│       └── news_clean.csv
│
├── models
│   ├── bow_vectorizer.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── word2vec.model
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── mlp.pkl
│   ├── X_train_*.pkl
│   ├── X_test_*.pkl
│   └── y_train/y_test.pkl
│
├── reports
│   ├── results.txt
│   ├── results.json
│   ├── report.md
│   ├── *_confusion_matrix.png
│   └── figures
│
├── src
│   ├── preprocess.py
│   ├── build_features.py
│   ├── train_models.py
│   └── evaluate.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

* Python 3
* pandas
* NumPy
* scikit-learn
* NLTK
* Gensim
* Matplotlib
* WordCloud
* Joblib

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Load Fake and True news datasets
* Assign class labels
* Merge datasets
* Shuffle records
* Convert text to lowercase
* Remove URLs
* Remove HTML tags
* Remove punctuation
* Remove numbers
* Remove extra whitespace
* Tokenize text
* Remove English stopwords
* Save cleaned dataset

---

### 2. Feature Engineering

Three text representations are generated:

* Bag of Words (BoW)
* TF-IDF
* Word2Vec Embeddings

The dataset is split into:

* 80% Training
* 20% Testing

using stratified sampling.

---

### 3. Models Implemented

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest
* Multi-Layer Perceptron (MLP)

Each trained model is saved for reuse.

---

### 4. Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Evaluation results are stored in:

* `reports/results.txt`
* `reports/results.json`

Confusion matrices are exported as PNG images.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd fake-news-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Data Preprocessing

```bash
python src/preprocess.py
```

---

### Step 2 — Feature Engineering

```bash
python src/build_features.py
```

---

### Step 3 — Train Models

```bash
python src/train_models.py
```

---

### Step 4 — Evaluate Models

```bash
python src/evaluate.py
```

---

## Output

After execution, the project generates:

* Cleaned dataset
* Feature vectors
* Trained machine learning models
* Evaluation metrics
* Confusion matrices
* Report template

---

## Future Improvements

Possible extensions include:

* BERT
* RoBERTa
* DistilBERT
* LSTM-based models
* Transformer architectures
* Explainable AI (XAI)
* Multilingual fake news detection
* Real-time news classification

---

## License

This project was created for academic and educational purposes as part of a summer internship.

---

## Acknowledgements

* Kaggle Fake and Real News Dataset
* scikit-learn
* NLTK
* Gensim
* Python Software Foundation
