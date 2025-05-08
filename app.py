
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")

uploaded_train = st.file_uploader("Upload Training CSV (xy_train.csv)", type="csv")
uploaded_test = st.file_uploader("Upload Testing CSV (x_test.csv)", type="csv")

if uploaded_train and uploaded_test:
    try:
        train_df = pd.read_csv(uploaded_train)
        test_df = pd.read_csv(uploaded_test)

        if 'text' in train_df.columns and 'label' in train_df.columns and 'text' in test_df.columns:
            st.success("Files uploaded successfully!")

            # Preprocess the data
            with st.spinner("Preprocessing text..."):
                train_df['text'] = train_df['text'].apply(preprocess)
                test_df['text'] = test_df['text'].apply(preprocess)

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(train_df['text'])
            y_train = train_df['label']
            X_test = vectorizer.transform(test_df['text'])

            # Train the model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Show accuracy if test has labels
            if 'label' in test_df.columns:
                y_test = test_df['label']
                predictions = model.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                st.success(f"Model trained. Accuracy on test data: {acc:.2f}")
            else:
                st.success("Model trained successfully!")

            # Message prediction
            st.subheader("Try Your Own News Headline")
            input_text = st.text_area("Enter a news headline or message:")
            if st.button("Predict"):
                clean = preprocess(input_text)
                vector = vectorizer.transform([clean])
                pred = model.predict(vector)[0]
                prob = model.predict_proba(vector)[0][1]
                st.markdown(f"**Prediction:** {'🟥 Fake' if pred == 1 else '🟩 Real'}")
                st.markdown(f"**Probability it's Fake:** {round(prob, 3)}")

        else:
            st.error("Train file must contain 'text' and 'label' columns. Test file must contain 'text'.")
    except Exception as e:
        st.error(f"Error reading files: {e}")
else:
    st.info("Please upload both the training and testing CSV files.")
