import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load Test Data
X_test_bow = joblib.load("models/X_test_bow.pkl")
X_test_tfidf = joblib.load("models/X_test_tfidf.pkl")
X_test_embed = joblib.load("models/X_test_embed.pkl")
y_test = joblib.load("models/y_test.pkl")

# Load Models

models = {
    "Logistic Regression": (
        joblib.load("models/logistic_regression.pkl"),
        X_test_tfidf,
    ),
    "KNN": (
        joblib.load("models/knn.pkl"),
        X_test_embed,
    ),
    "Random Forest": (
        joblib.load("models/random_forest.pkl"),
        X_test_bow,
    ),
    "Neural Network": (
        joblib.load("models/mlp.pkl"),
        X_test_tfidf,
    ),
}

results = {}

text_output = []

for name, (model, X) in models.items():

    print(f"Evaluating {name}...")

    predictions = model.predict(X)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1 Score": round(float(f1), 4),
    }

    text_output.append(f"{name}")
    text_output.append("-" * 40)
    text_output.append(f"Accuracy : {accuracy:.4f}")
    text_output.append(f"Precision: {precision:.4f}")
    text_output.append(f"Recall   : {recall:.4f}")
    text_output.append(f"F1 Score : {f1:.4f}")
    text_output.append("")

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"],
    )

    disp.plot()

    plt.savefig(
        f"reports/{name.replace(' ','_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

with open("reports/results.json", "w") as file:
    json.dump(results, file, indent=4)

with open("reports/results.txt", "w") as file:
    file.write("\n".join(text_output))

print("\nEvaluation Complete.")