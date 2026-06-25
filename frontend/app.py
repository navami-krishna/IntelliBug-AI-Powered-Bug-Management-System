import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Bug Management System",
    page_icon="🐞",
    layout="wide"
)
if "show_home" not in st.session_state:
    st.session_state.show_home = True
if st.session_state.show_home:

    # Light background color
    st.markdown("""
    <style>
    .stApp {
        background-color: #2E5894;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add some space from top
    st.write("")
    st.write("")
    st.write("")

    # Center and enlarge logo
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image("logo.png", width=650)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.show_home = False
            st.rerun()

    st.stop()



    
# ---------------- REMOVE TOP GAP ----------------


st.markdown("""
<style>
/* Reduce top spacing */
.block-container {
    padding-top: 4rem;
}

/* Center title text */
.center-title {
    text-align: center;
    margin-top: 5px;
    margin-bottom: 5px;
    font-size: 40px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)
  
if st.sidebar.button("🏠 Home"):
    st.session_state.show_home = True
    st.rerun()

st.markdown(
    "<div class='center-title'>🐞 IntelliBug AI System</div>",
    unsafe_allow_html=True
)



# ---------------- SPACING ----------------
st.markdown("<br>", unsafe_allow_html=True)

# ---------------- MAIN BANNER (KEEP ONLY THIS ONE) ----------------

# ---------------- MODERN DARK-LIGHT BLUE THEME ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #dbe9f4, #eef3f8);
    color: #1e2a38;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f2a44, #2c3e50);
    color: white;
}

/* Sidebar text */
div[data-testid="stSidebar"] * {
    color: white;
}

/* Main Title */
h1 {
    color: #1f2a44;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    background-color: #2c3e50;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton>button:hover {
    background-color: #1f2a44;
}

/* Text area */
.stTextArea textarea {
    border-radius: 10px;
    border: 1px solid #b8c7d6;
}

/* Info/Success boxes */
.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER BANNER ----------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f2a44, #2c3e50);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 23px;
">
    <h1 style="margin:0; font-size:28px;">
        🐞 Intelligent Bug Management Platform
    </h1>
    <p style="margin:5px 0 0 0; opacity:0.85;">
        Predict severity, resolution, category, fix time, recommend developers, detect duplicates & explain predictions
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
module = st.sidebar.selectbox(
    "Select Module",
    [
        "Severity Prediction",
        "Resolution Prediction",
        "Category Prediction",
        "Fix Time Category Prediction",
        "Developer Recommendation",
        "Duplicate Detection",
        "Explainable AI"
    ]
)


bug_text = st.text_area("Enter Bug Report")

# ---------------- SEVERITY ----------------
if module == "Severity Prediction":
    if st.button("Predict Severity"):

        model = joblib.load("../model/best_severity_model.pkl")
        tfidf = joblib.load("../model/severity_tfidf.pkl")

        vector = tfidf.transform([bug_text])
        prediction = model.predict(vector)[0]

        st.success(f"Predicted Severity: {prediction}")

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(vector)[0]) * 100
            st.info(f"Confidence: {confidence:.2f}%")
        
           

# ---------------- RESOLUTION ----------------
elif module == "Resolution Prediction":
    if st.button("Predict Resolution"):

        model = joblib.load("../model/best_resolution_model.pkl")
        tfidf = joblib.load("../model/resolution_tfidf.pkl")

        vector = tfidf.transform([bug_text])
        prediction = model.predict(vector)[0]

        st.success(f"Predicted Resolution: {prediction}")
        

# ---------------- CATEGORY ----------------
elif module == "Category Prediction":
    if st.button("Predict Category"):

        model = joblib.load("../model/best_category_model.pkl")
        tfidf = joblib.load("../model/category_tfidf.pkl")

        vector = tfidf.transform([bug_text])
        prediction = model.predict(vector)[0]

        st.success(f"Predicted Category: {prediction}")
        
    

# ---------------- FIX TIME ----------------
elif module == "Fix Time Category Prediction":
    if st.button("Predict Fix Time"):

        model = joblib.load("../model/fix_time_category_model.pkl")
        tfidf = joblib.load("../model/fix_time_category_tfidf.pkl")

        vector = tfidf.transform([bug_text])
        prediction = model.predict(vector)[0]

        st.warning(f"Estimated Fix Time Category: {prediction}")
       
    

# ---------------- DEVELOPER ----------------
elif module == "Developer Recommendation":
    if st.button("Recommend Developer"):

        model = joblib.load("../model/best_developer_model.pkl")
        tfidf = joblib.load("../model/developer_tfidf.pkl")

        vector = tfidf.transform([bug_text])
        prediction = model.predict(vector)[0]

        st.success(f"Recommended Developer: {prediction}")
    
   

# ---------------- DUPLICATE ----------------
elif module == "Duplicate Detection":
    if st.button("Check Duplicate"):

        duplicate_system = joblib.load("../model/duplicate_detection.pkl")

        tfidf = duplicate_system["tfidf"]
        X_tfidf = duplicate_system["matrix"]
        df = duplicate_system["data"]

        vector = tfidf.transform([bug_text])

        similarity = cosine_similarity(vector, X_tfidf)

        max_index = similarity.argmax()
        max_score = similarity[0][max_index]

        st.write(f"Similarity Score: {max_score*100:.2f}%")

        if max_score >= 0.35:
            st.error("Duplicate Bug Found!")
            st.write("Bug ID:", df.iloc[max_index]["bug_id"])
            st.write("Description:", df.iloc[max_index]["short_description"])
        else:
            st.success("No Duplicate Bug Found")
           


# ---------------- EXPLAINABLE AI ----------------
elif module == "Explainable AI":
    if st.button("Explain Prediction"):

        model = joblib.load("../model/best_severity_model.pkl")
        tfidf = joblib.load("../model/severity_tfidf.pkl")

        vector = tfidf.transform([bug_text])
        prediction = model.predict(vector)[0]

        st.success(f"Predicted Severity: {prediction}")

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(vector)[0]) * 100
            st.info(f"Confidence: {confidence:.2f}%")

        feature_names = tfidf.get_feature_names_out()
        scores = vector.toarray()[0]

        top_indices = np.argsort(scores)[::-1][:10]

        st.subheader("Top Words Influencing Prediction")

        for i in top_indices:
            if scores[i] > 0:
                st.write(f"• {feature_names[i]}")
            
   