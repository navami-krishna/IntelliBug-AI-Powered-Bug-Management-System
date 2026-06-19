import joblib
import numpy as np

# Load Saved Model

model = joblib.load(
    "model/best_severity_model.pkl"
)

tfidf = joblib.load(
    "model/severity_tfidf.pkl"
)

print("Explainable AI Module Ready")

while True:

    bug = input(
        "\nEnter Bug Report (type exit to stop): "
    )

    if bug.lower() == "exit":
        break

    # Convert Text

    bug_vector = tfidf.transform([bug])

    # Prediction

    prediction = model.predict(
        bug_vector
    )

    print(
        "\nPredicted Severity:",
        prediction[0]
    )

    # Confidence

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            bug_vector
        )

        print(
            "Confidence:",
            round(
                max(probabilities[0]) * 100,
                2
            ),
            "%"
        )

    # Important Words

    feature_names = (
        tfidf.get_feature_names_out()
    )

    tfidf_scores = (
        bug_vector.toarray()[0]
    )

    top_indices = np.argsort(
        tfidf_scores
    )[::-1][:10]

    print(
        "\nTop Words Influencing Prediction:"
    )

    for i in top_indices:

        if tfidf_scores[i] > 0:

            print(
                f"{feature_names[i]} "
                f"({round(tfidf_scores[i],4)})"
            )