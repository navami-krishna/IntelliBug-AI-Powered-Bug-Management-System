import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print(classification_report(y_test, y_pred))


# User Testing
while True:

    bug = input("\nEnter Bug Report (type exit to stop): ")

    if bug.lower() == "exit":
        break

    bug_vector = tfidf.transform([bug])

    prediction = model.predict(bug_vector)

    probabilities = model.predict_proba(bug_vector)

    print("Predicted Category:", prediction[0])

    print(
        "Confidence:",
        round(max(probabilities[0]) * 100, 2),
        "%"
    )