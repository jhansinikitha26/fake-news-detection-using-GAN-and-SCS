Fake News Detection using GAN and Source Credibility Scoring (SCS)
This project combines Generative Adversarial Networks (GANs) with Source Credibility Scoring (SCS) to detect fake news articles with improved accuracy. The model processes real and fake news datasets, applies text preprocessing and TF-IDF vectorization, and integrates a GAN-based generator and discriminator for learning content patterns. Additionally, credibility is assigned based on domain sources to weigh predictions more accurately.

 Features
Preprocessing and cleaning of real and fake news datasets

TF-IDF vectorization of news content

GAN architecture to generate and classify news articles

Source Credibility Scoring based on domain reputation

Final prediction based on GAN + SCS weighted combination

Random Forest baseline for performance comparison

Explainable predictions using LIME

Technologies Used
Python

Pandas, NumPy

TensorFlow / Keras

Scikit-learn

NLTK (text preprocessing)

tldextract (domain parsing)

LIME (explainability)

 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/fake-news-detection-gan-scs.git
cd fake-news-detection-gan-scs
Install the required libraries:

nginx
Copy
Edit
pip install -r requirements.txt
Unzip and place fake.csv and true.csv datasets inside the /data folder or specify the correct path.

▶ Usage
Run the main notebook or Python script in Google Colab or Jupyter Notebook.

The code will:

Load and clean data

Apply TF-IDF

Train a GAN and Discriminator

Score based on source credibility

Combine the outputs for final classification

Evaluate results using both GAN+SCS and Random Forest

Use LIME for interpreting prediction on sample articles

 Results
GAN + SCS improves detection by combining content-based signals with external trustworthiness.

Accuracy using Random Forest: ~93%

LIME explanations help identify which words contribute most to predictions.
