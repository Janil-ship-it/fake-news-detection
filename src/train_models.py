import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

print("Loading datasets...")

X_train_bow = joblib.load("models/X_train_bow.pkl")
X_train_tfidf = joblib.load("models/X_train_tfidf.pkl")
X_train_embed = joblib.load("models/X_train_embed.pkl")
y_train = joblib.load("models/y_train.pkl")

# Logistic Regression
print("Training Logistic Regression")

logistic = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic.fit(X_train_tfidf, y_train)

joblib.dump(
    logistic,
    "models/logistic_regression.pkl"
)

print("Logistic Regression Saved")

# KNN
print("Training KNN")

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_embed,
    y_train
)

joblib.dump(
    knn,
    "models/knn.pkl"
)

print("KNN Saved")

# Random Forest
print("Training Random Forest")

forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

forest.fit(
    X_train_bow,
    y_train
)

joblib.dump(
    forest,
    "models/random_forest.pkl"
)

print("Random Forest Saved")

# Neural Network
print("Training Neural Network")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=20,
    random_state=42
)

mlp.fit(
    X_train_tfidf,
    y_train
)

joblib.dump(
    mlp,
    "models/mlp.pkl"
)
print("All models trained successfully")