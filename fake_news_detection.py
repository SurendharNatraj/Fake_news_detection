
# Step 1: Install Required Libraries (Uncomment if needed)
# !pip install pandas scikit-learn nltk

# Step 2: Import Libraries and Download NLTK Data
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

# Step 3: Load Your CSV Files
train_df = pd.read_csv("xy_train.csv")
test_df = pd.read_csv("x_test.csv")

# Step 4: Preprocessing Function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Step 5: Apply Preprocessing
train_df['text'] = train_df['text'].apply(preprocess)
test_df['text'] = test_df['text'].apply(preprocess)

# Step 6: Vectorize and Train Model
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['text'])

model = LogisticRegression()
model.fit(X_train, y_train)

# Optional: Evaluate Model (if test labels available)
if 'label' in test_df.columns:
    y_test = test_df['label']
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Test with Custom Message
def detect_message(message):
    clean = preprocess(message)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]
    print(f"Message: {message}")
    print("Prediction:", "Fake" if prediction == 1 else "Real")
    print("Probability it's fake:", round(probability, 3))

# Step 8: Try it
detect_message("Breaking: Scientists discover water on Mars!")
