import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("data/eclipse_bug_report_data_balanced.csv")
print(df["severity_category"].value_counts())

# Create text column
df["text"] = (
    df["short_description"].fillna("") +
    " " +
    df["long_description"].fillna("")
)

# Features and Target
X = df["text"]
y = df["severity_category"]

# Convert text to numbers
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Unique Predictions:")
print(set(y_pred))

print(
    "Naive Bayes Accuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)
print(classification_report(y_test, y_pred))

while True:

    bug = input("\nEnter Bug Report (type exit to stop): ")

    if bug.lower() == "exit":
        break

    bug_vector = tfidf.transform([bug])

    prediction = model.predict(bug_vector)

    probabilities = model.predict_proba(bug_vector)

    print("Predicted Severity:", prediction[0])

    print(
        "Confidence:",
        round(max(probabilities[0]) * 100, 2),
        "%"
    )