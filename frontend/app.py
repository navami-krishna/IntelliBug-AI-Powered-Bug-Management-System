import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Bug Management System",
    page_icon="🐞",
    layout="wide"
)

st.title("🐞 AI Bug Management System")

# Sidebar

module = st.sidebar.selectbox(
    "Select Module",
    [
        "Severity Prediction",
        "Resolution Prediction",
        "Category Prediction",
        "Fix Time Prediction",
        "Developer Recommendation",
        "Duplicate Detection"
    ]
)

bug_text = st.text_area(
    "Enter Bug Report"
)

# --------------------------------
# Severity
# --------------------------------

if module == "Severity Prediction":

    if st.button("Predict Severity"):

        model = joblib.load(
            "model/best_severity_model.pkl"
        )

        tfidf = joblib.load(
            "model/severity_tfidf.pkl"
        )

        vector = tfidf.transform(
            [bug_text]
        )

        prediction = model.predict(
            vector
        )[0]

        st.success(
            f"Predicted Severity: {prediction}"
        )

        if hasattr(model, "predict_proba"):

            confidence = max(
                model.predict_proba(
                    vector
                )[0]
            ) * 100

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

# --------------------------------
# Resolution
# --------------------------------

elif module == "Resolution Prediction":

    if st.button("Predict Resolution"):

        model = joblib.load(
            "model/best_resolution_model.pkl"
        )

        tfidf = joblib.load(
            "model/resolution_tfidf.pkl"
        )

        vector = tfidf.transform(
            [bug_text]
        )

        prediction = model.predict(
            vector
        )[0]

        st.success(
            f"Predicted Resolution: {prediction}"
        )

# --------------------------------
# Category
# --------------------------------

elif module == "Category Prediction":

    if st.button("Predict Category"):

        model = joblib.load(
            "model/best_category_model.pkl"
        )

        tfidf = joblib.load(
            "model/category_tfidf.pkl"
        )

        vector = tfidf.transform(
            [bug_text]
        )

        prediction = model.predict(
            vector
        )[0]

        st.success(
            f"Predicted Category: {prediction}"
        )

# --------------------------------
# Fix Time
# --------------------------------

elif module == "Fix Time Prediction":

    if st.button("Predict Fix Time"):

        model = joblib.load(
            "model/best_fix_time_model.pkl"
        )

        tfidf = joblib.load(
            "model/fix_time_tfidf.pkl"
        )

        vector = tfidf.transform(
            [bug_text]
        )

        prediction = model.predict(
            vector
        )[0]

        st.success(
            f"Estimated Fix Time: {round(prediction,2)} Days"
        )

# --------------------------------
# Developer Recommendation
# --------------------------------

elif module == "Developer Recommendation":

    if st.button("Recommend Developer"):

        model = joblib.load(
            "model/best_developer_model.pkl"
        )

        tfidf = joblib.load(
            "model/developer_tfidf.pkl"
        )

        vector = tfidf.transform(
            [bug_text]
        )

        prediction = model.predict(
            vector
        )[0]

        st.success(
            f"Recommended Developer: {prediction}"
        )

# --------------------------------
# Duplicate Detection
# --------------------------------

elif module == "Duplicate Detection":

    if st.button("Check Duplicate"):

        duplicate_system = joblib.load(
            "model/duplicate_detection.pkl"
        )

        tfidf = duplicate_system["tfidf"]
        X_tfidf = duplicate_system["matrix"]
        df = duplicate_system["data"]

        vector = tfidf.transform(
            [bug_text]
        )

        similarity = cosine_similarity(
            vector,
            X_tfidf
        )

        max_index = similarity.argmax()

        max_score = similarity[0][max_index]

        st.write(
            f"Similarity Score: {max_score*100:.2f}%"
        )

        if max_score >= 0.35:

            st.error(
                "Duplicate Bug Found!"
            )

            st.write(
                "Bug ID:",
                df.iloc[max_index]["bug_id"]
            )

            st.write(
                "Description:",
                df.iloc[max_index]["short_description"]
            )

        else:

            st.success(
                "No Duplicate Bug Found"
            )