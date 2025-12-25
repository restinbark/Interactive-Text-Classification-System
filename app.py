#Streamlit for web app

import streamlit as st   
from dataset import create_dataset

st.set_page_config(page_title="Text Classification Project", layout="wide")

st.title("📘 Interactive Text Classification System")

st.sidebar.title("ML Pipeline")
page = st.sidebar.radio(
    "Select a Step",
        ["Dataset Overview", "Text Preprocessing","TF-IDF Vectorization","Naive Bayes Model","Prediction","Model Evaluation","Results & Analysis",
        "Run Full Pipeline"
    ]
)

if page == "Dataset Overview":
    st.header("📂 Dataset Preparation")

    df = create_dataset()

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Class Distribution")
    st.write(df["label"].value_counts())

    st.markdown("""
    **Corpus Definition:**  
    A corpus is a collection of text documents used for training a machine learning model.
    """)

#Text preprocessing function

from utils import preprocess_text    

if page == "Text Preprocessing":
    st.header("🧹 Text Preprocessing")

    user_text = st.text_area("Enter text to preprocess:")

    if user_text:
        processed_text = preprocess_text(user_text)

        st.subheader("Original Text")
        st.write(user_text)

        st.subheader("Processed Text")
        st.write(processed_text)

#TF-IDF vectorizer function

from vectorizer import build_vectorizer  

if page == "TF-IDF Vectorization":
    st.header("📐 TF-IDF Vectorization")

    df = create_dataset()

    vectorizer, X = build_vectorizer(df["text"])

    st.subheader("TF-IDF Feature Matrix")
    st.write(f"Shape of TF-IDF matrix: {X.shape}")

    st.subheader("Vocabulary Size")
    st.write(len(vectorizer.vocabulary_))

    sample_text = st.text_input("Enter text to convert into TF-IDF vector:")

    if sample_text:
        sample_vector = vectorizer.transform([sample_text])
        st.subheader("Vector Shape for Input Text")
        st.write(sample_vector.shape)

        st.markdown("""
        **Note:**  
        Words not seen during training are ignored by the TF-IDF vectorizer.
        """)

#Model training function

from model import train_model   

if page == "Naive Bayes Model":
    st.header("🤖 Naive Bayes Model Training")

    df = create_dataset()

    vectorizer, X = build_vectorizer(df["text"])
    y = df["label"]

    if st.button("Train Model"):
        model = train_model(X, y)
        st.success("Model trained successfully!")

        st.subheader("Model Details")
        st.write("Algorithm: Multinomial Naive Bayes")
        st.write(f"Number of classes: {len(model.classes_)}")
        st.write("Classes:", model.classes_)

#Prediction interface

if page == "Prediction":    
    st.header("🔮 Text Classification Prediction")

    df = create_dataset()
    vectorizer, X = build_vectorizer(df["text"])
    y = df["label"]
    model = train_model(X, y)

    user_input = st.text_area("Enter text to classify:")

    if st.button("Predict") and user_input:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Category: {prediction}")

        st.subheader("Confidence Scores")
        for label, prob in zip(model.classes_, probabilities):
            st.write(f"{label}: {prob:.2f}")


#Model evaluation 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import pandas as pd

if page == "Model Evaluation":
    st.header("📊 Model Evaluation")

    df = create_dataset()
    vectorizer, X = build_vectorizer(df["text"])
    y = df["label"]
    model = train_model(X, y)

    # Predictions on training data
    y_pred = model.predict(X)

    st.subheader("Accuracy")
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")

    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    st.dataframe(cm_df)

    st.markdown("""
    **Note:**  
    The evaluation is performed on the training dataset for demonstration purposes.
    """)


#Results and analysis

if page == "Results & Analysis":  
    st.header("🧠 Results & Analysis")

    st.subheader("Example Predictions")
    examples = [
        "walking daily improves physical health",
        "students learn through education",
        "meditation helps inner peace",
        "barikilign is a driver"
    ]

    df = create_dataset()
    vectorizer, X = build_vectorizer(df["text"])
    model = train_model(X, df["label"])

    for text in examples:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = max(probs)

        st.write(f"**Text:** {text}")
        st.write(f"Prediction: {pred} (confidence: {confidence:.2f})")
        st.markdown("---")

    st.subheader("Analysis of Misclassified or Low-Confidence Samples")
    st.markdown("""
    - Some inputs do not strongly belong to any category.
    - The model shows low confidence in such cases.
    - This behavior is expected with a small, domain-specific dataset.
    """)

    st.subheader("Model Behavior Insights")
    st.markdown("""
    - TF-IDF ignores unseen words.
    - Naive Bayes predicts the most probable class.
    - Confidence scores reflect uncertainty.
    """)


#Run full pipeline


if page == "Run Full Pipeline":
    st.header("🚀 Run Full ML Pipeline")

    df = create_dataset()

    st.write("Step 1: Dataset loaded")
    st.dataframe(df.head())

    vectorizer, X = build_vectorizer(df["text"])
    st.success("Step 2: TF-IDF vectorization completed")

    model = train_model(X, df["label"])
    st.success("Step 3: Naive Bayes model trained")

    user_text = st.text_area("Enter text to run through full pipeline:")

    if st.button("Run Pipeline") and user_text:
        vec = vectorizer.transform([user_text])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = max(probs)

        st.subheader("Final Output")
        st.write(f"Prediction: **{pred}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        if confidence < 0.5:
            st.warning("Low confidence prediction. Input may not strongly match any category.")




