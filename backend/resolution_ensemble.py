import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset

df = pd.read_csv("data/mozilla_bug_report_data_balanced.csv")

df = df.dropna(subset=["resolution_category"])
df["resolution_category"] = df["resolution_category"].astype(str)

df = df[
    (df["resolution_category"] != "0") &
    (df["resolution_category"] != "nan")
]

print(df["resolution_category"].value_counts())
print(df["resolution_category"].unique())

# Create text column

df["text"] = (
    df["short_description"].fillna("")
    + " "
    + df["long_description"].fillna("")
)

# Features and Target

X = df["text"]
y = df["resolution_category"]

print("NaN in y:", y.isna().sum())

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

model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('nb', MultinomialNB()),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    voting='hard'

)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

print("Unique Predictions:")
print(set(y_pred))

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

    print("Predicted Resolution:", prediction[0])

    print(
        "Confidence:",
        round(max(probabilities[0]) * 100, 2),
        "%"
    )