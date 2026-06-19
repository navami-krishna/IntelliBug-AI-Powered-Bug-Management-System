import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Dataset

df = pd.read_csv("data/Bug Reports Dataset.csv")

# Remove Missing Categories
df = df.dropna(subset=["Category"])

# Create Text Column
df["text"] = (
    df["Product"].fillna("") + " " +
    df["Component"].fillna("") + " " +
    df["Summary"].fillna("")
)

# Features and Target
X = df["text"]
y = df["Category"]

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = tfidf.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

    "SVM":
        SVC(
            kernel="linear",
            probability=True,
            random_state=42
        ),

    "Ensemble Voting":
        VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                ),
                (
                    "nb",
                    MultinomialNB()
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ],
            voting="hard"
        )
}

import joblib

# Find Best Model
results=[]
best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in model.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append([
        name,
        round(accuracy * 100, 2),
        round(precision * 100, 2),
        round(recall * 100, 2),
        round(f1 * 100, 2)
    ])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Save Best Model

joblib.dump(
    best_model,
    "model/best_category_model.pkl"
)

joblib.dump(
    tfidf,
    "model/category_tfidf.pkl"
)

print("\nBest Model Saved Successfully")
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)"
    ]
)
print("Best Model:", best_model_name)
print("\n" + "=" * 70)
print("RESOLUTION MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

print("\n" + "=" * 70)

best_model = results_df.loc[
    results_df["Accuracy (%)"].idxmax()
]
print(
    "Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)