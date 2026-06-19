import pandas as pd
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset

df = pd.read_csv("data/mozilla_bug_report_data.csv")
# Remove Missing Assignees

df = df.dropna(subset=["assignee_name"])
df = df[df["assignee_name"] != "nobody"]

# Optional: Keep Top 20 Developers
top_assignees = df["assignee_name"].value_counts().head(20).index

df = df[
    df["assignee_name"].isin(top_assignees)
]

# Create Text Column

df["text"] = (
    df["product_name"].fillna("")
    + " "
    + df["component_name"].fillna("")
    + " "
    + df["short_description"].fillna("")
    + " "
    + df["long_description"].fillna("")
)

# Features and Target

X = df["text"]
y = df["assignee_name"]

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

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, y_pred) * 100,
        2
    ),
    "%"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# User Testing

while True:

    bug = input(
        "\nEnter Bug Report (type exit to stop): "
    )

    if bug.lower() == "exit":
        break

    bug_vector = tfidf.transform([bug])

    prediction = model.predict(
        bug_vector
    )

    probabilities = model.predict_proba(
        bug_vector
    )

    print(
        "Recommended Developer:",
        prediction[0]
    )

    print(
        "Confidence:",
        round(
            max(probabilities[0]) * 100,
            2
        ),
        "%"
    )