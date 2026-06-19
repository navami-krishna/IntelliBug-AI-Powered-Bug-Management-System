import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset

df = pd.read_csv("data/mozilla_bug_report_data.csv")

# Create Text Column

df["text"] = (
    df["short_description"].fillna("")
    + " "
    + df["long_description"].fillna("")
)
print(df.iloc[0]["text"])

# TF-IDF

tfidf = TfidfVectorizer(
    stop_words="english",
    
)

X_tfidf = tfidf.fit_transform(df["text"])

# Save TF-IDF
duplicate_system = {
    "tfidf": tfidf,
    "matrix": X_tfidf,
    "data": df
}

joblib.dump(
    duplicate_system,
    "model/duplicate_detection.pkl"
)

print("Duplicate Detection System Saved Successfully")
duplicate_system = joblib.load(
    "model/duplicate_detection.pkl"
)


joblib.dump(
    tfidf,
    "model/duplicate_tfidf.pkl"
)

print("Duplicate Detection System Ready")

# User Testing

while True:

    bug = input(
        "\nEnter Bug Report (type exit to stop): "
    )

    if bug.lower() == "exit":
        break

    # Convert New Bug

    bug_vector = tfidf.transform([bug])

    # Similarity

    similarities = cosine_similarity(
        bug_vector,
        X_tfidf
    )
    

    max_index = similarities.argmax()

    max_score = similarities.max()

    print("Similarity Score:", max_score)

    
    print("Highest Similarity:", max_score)

    if max_score >= 0.35:

        print("\nDuplicate Bug Found!")

        print(
            "\nExisting Bug ID:",
            df.iloc[max_index]["bug_id"]
        )

        print(
            "\nShort Description:"
        )

        print(
            df.iloc[max_index]["short_description"]
        )

    else:

        print(
            "\nNo Duplicate Bug Found"
        )