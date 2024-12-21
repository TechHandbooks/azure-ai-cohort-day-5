import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load 20 Newsgroups dataset
categories = ['sci.med', 'rec.autos']  # Select two categories for simplicity
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Create a DataFrame
df = pd.DataFrame({'text': data.data, 'target': data.target})
print(df.head())

# Map target to category names for better readability
df['category'] = df['target'].map(lambda x: data.target_names[x])
print(df.head())


# Initialize tools
stop_words = set(word.lower() for word in stopwords.words('english'))  # Ensure stopwords are lowercase strings
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())  # Tokenization
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stemming
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]  # Lemmatization
    return " ".join(lemmatized_tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)
print(df.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['target'], test_size=0.2, random_state=42)


# Convert text data to numerical feature vectors
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.metrics import accuracy_score, classification_report

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))







