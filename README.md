# fake-news-detection-using-GAN-and-SCS
from google.colab import files
uploaded = files.upload()

from zipfile import ZipFile
# Unzip fake.csv
with ZipFile("fake.csv (1).zip", 'r') as zip_ref:
    zip_ref.extractall("/content/fake_news")
# Unzip true.csv
with ZipFile("true.csv.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/true_news")

import pandas as pd
# Load the CSVs
fake_df = pd.read_csv('/content/fake_news/fake.csv')
true_df = pd.read_csv('/content/true_news/true.csv')
# Add labels: 1 = Fake, 0 = Real
fake_df['label'] = 1
true_df['label'] = 0
# Combine datasets
data = pd.concat([fake_df, true_df], ignore_index=True)
# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# Check basic info
print("Shape of dataset:", data.shape)
data.head()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # Remove punctuation and numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)
# Apply to the text column (combine title + text if both exist)
if 'title' in data.columns and 'text' in data.columns:
    data['content'] = data['title'].fillna('') + " " + data['text'].fillna('')
else:
    data['content'] = data['text']
data['clean_content'] = data['content'].apply(clean_text)
# Preview cleaned data
data[['label', 'clean_content']].head()

from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_content']).toarray()
# Labels
y = data['label'].values
print("TF-IDF shape:", X.shape)

import tldextract
# Example: A simple source credibility scoring function
# You can improve this with more sophisticated logic based on your dataset
def extract_source(text):
    # Try to extract the domain from the source text
    try:
        extracted = tldextract.extract(text)
        return extracted.domain
    except:
        return ""
def assign_credibility_score(domain):
    credible_sources = ['bbc', 'reuters', 'nytimes', 'guardian', 'cnn']
    if any(cred in domain for cred in credible_sources):
        return 1  # Credible source
    else:
        return 0  # Non-credible source
# Assuming 'source' column exists, if not we simulate with domain info.
data['source'] = data['content'].apply(lambda x: extract_source(x))
# Apply credibility scoring
data['credibility'] = data['source'].apply(assign_credibility_score)
# Preview the data with added source and credibility
data[['label', 'content', 'source', 'credibility']].head()

!pip install tldextract

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
# Set random seed for reproducibility
tf.random.set_seed(42)
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Generator Model (Fake News Generator)
def build_generator(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(input_dim, activation='sigmoid'))
    return model
# Discriminator Model (Fake News Classifier)
def build_discriminator(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
# Build GAN model (combined Generator + Discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model
# Initialize generator and discriminator
generator = build_generator(input_dim=X_train.shape[1])
discriminator = build_discriminator(input_dim=X_train.shape[1])
# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile GAN model
gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')
# Summary of the models
generator.summary()
discriminator.summary()

import numpy as np
# Training function
def train_gan(generator, discriminator, gan, X_train, y_train, epochs=1000, batch_size=64):
    for epoch in range(epochs):
        # 1. Train the Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]
        real_labels = y_train[idx]
        fake_data = generator.predict(np.random.randn(batch_size, X_train.shape[1]))
        fake_labels = np.zeros(batch_size)  # Fake news has label 0
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # 2. Train the Generator
        noise = np.random.randn(batch_size, X_train.shape[1])
        valid_labels = np.ones(batch_size)  # Generator tries to make fake data appear rea
        g_loss = gan.train_on_batch(noise, valid_labels)
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
# Start training the GAN
train_gan(generator, discriminator, gan, X_train, y_train, epochs=1000, batch_size=64)

# Assuming we have a credibility score function for each article
def get_credibility_score(article):
    # Simple example: You can expand this with more sophisticated logic
    # If the source is from a high-credibility domain (example: cnn.com, bbc.com, etc.), assign a high score
    high_cred_sources = ['cnn.com', 'bbc.com', 'nytimes.com']
    # Example: Extract the source domain from the URL (you can also use other source metadata)
    domain = tldextract.extract(article)
    source = domain.domain + '.' + domain.suffix
    if source in high_cred_sources:
        return 1  # High credibility
    else:
        return 0  # Low credibility
# Example final prediction logic combining both
def final_prediction(discriminator_output, article):
    credibility_score = get_credibility_score(article)

    # Weights: Give more weight to content if the source is credible
    if credibility_score == 1:
        final_decision = discriminator_output * 0.7 + credibility_score * 0.3  # More focus on discriminator for credible sources
    else:
        final_decision = discriminator_output * 0.5 + credibility_score * 0.5  # Equal weight for fake and non-credible sources
    # If final decision is closer to 1, classify as real (real news), if closer to 0, classify as fake
    return 1 if final_decision > 0.5 else 0

# Example: Using the trained discriminator to predict and then combining with SCS
def predict_fake_news_with_scs(article_text):
    # Step 1: Use the discriminator model to classify the article as real (1) or fake (0)
    article_vector = tfidf.transform([article_text]).toarray()  # Transform the text to TF-IDF features
    discriminator_output = discriminator.predict(article_vector)
    # Step 2: Combine with SCS to make final decision
    final_decision = final_prediction(discriminator_output, article_text)
    return final_decision
# Test with an example article
example_article = "This is a sample news article about recent political developments."
prediction = predict_fake_news_with_scs(example_article)
print("Final Prediction (1 = Real, 0 = Fake):", prediction)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", sum(y_train==1), "Fake,", sum(y_train==0), "Real")
print("Test :", sum(y_test==1), "Fake,", sum(y_test==0), "Real")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


import joblib
joblib.dump(clf, 'fake_news_model.pkl')

!pip install lime --quiet
from lime.lime_text import LimeTextExplainer
import numpy as np
# Create a prediction pipeline
class_names = ['Real', 'Fake']
# Since we used TF-IDF to train, we need the vectorizer and classifier
def predict_proba(texts):
    vectors = tfidf.transform(texts)
    return clf.predict_proba(vectors)
explainer = LimeTextExplainer(class_names=class_names)
# Example: Predict and explain one article
sample_text = "Breaking: The government has confirmed a UFO landing in Nevada."
# Show prediction
pred = clf.predict(tfidf.transform([sample_text]))[0]
print(f"Prediction: {'Fake' if pred == 1 else 'Real'}")
# Explanation
exp = explainer.explain_instance(sample_text, predict_proba, num_features=10)
exp.show_in_notebook(text=True)


