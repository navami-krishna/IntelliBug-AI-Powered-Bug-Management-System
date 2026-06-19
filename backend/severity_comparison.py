import pandas as pd

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
df = pd.read_csv("data/eclipse_bug_report_data_balanced.csv")


# Create Text Column
df["text"] = (
    df["short_description"].fillna("")
    + " "
    + df["long_description"].fillna("")
)

# Features and Target
X = df["text"]
y = df["severity_category"]

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = tfidf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

results = []

best_accuracy = 0
best_model = None
best_model_name = ""
models = {
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
                ("lr", LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )),
                ("nb", MultinomialNB()),
                ("rf", RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                ))
            ],
            voting="hard"
        )
}
for name, model in models.items():

    print(f"\nTraining {name}...")

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
import joblib
joblib.dump(best_model,"model/best_severity_model.pkl")

joblib.dump(tfidf,"model/severity_tfidf.pkl")

print("\nBest Model Saved Successfully")



# ---------------------------------
# Display Results
# ---------------------------------

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

print("\n" + "=" * 70)
print("RESOLUTION MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

print("\n" + "=" * 70)

best_model = results_df.loc[
    results_df["Accuracy (%)"].idxmax()
]

print("BEST MODEL")
print("=" * 70)

print(best_model)