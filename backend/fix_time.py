
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("data/mozilla_bug_report_data_balanced.csv")

# Remove missing values
df = df.dropna(subset=["bug_fix_time"])

# Create text column
df["text"] = (
    df["short_description"].fillna("")
    + " "
    + df["long_description"].fillna("")
)

# Features and Target
X = df["text"]
y = df["bug_fix_time"]
print(df["bug_fix_time"].head())

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = tfidf.fit_transform(X)
print(X_tfidf)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf,y,test_size=0.2,random_state=42)

# Model
print(X_train)
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2), "days")
print("R2 Score:", round(r2, 2))

# Save Model
import joblib

joblib.dump(model, "model/fix_time_model.pkl")
joblib.dump(tfidf, "model/fix_time_tfidf.pkl")

print("Fix Time Model Saved Successfully")

# User Testing
while True:

    bug = input("\nEnter Bug Report (type exit to stop): ")

    if bug.lower() == "exit":
        break

    bug_vector = tfidf.transform([bug])

    prediction = model.predict(bug_vector)

    print(
        "Estimated Fix Time:",
        round(prediction[0], 2),
        "days"
    )
    