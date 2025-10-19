import pandas as pd
import numpy as np
import re, string, nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Loading dataset...")

# Read CSV safely
df = pd.read_csv("complaints.csv", on_bad_lines='skip', engine='python')

# Keep only necessary columns
df = df[['Product', 'Consumer complaint narrative']]
df.dropna(inplace=True)
print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# Correct category names (as seen from your dataset)
categories = [
    'Credit reporting, credit repair services, or other personal consumer reports',
    'Debt collection',
    'Consumer Loan',
    'Mortgage'
]

# Filter only selected categories
df = df[df['Product'].isin(categories)]
print(f"✅ Data after filtering: {df.shape}")

if df.empty:
    raise ValueError("⚠️ Filtered dataset is empty! Check your category names again.")

# Map category labels to numbers
label_map = {cat: i for i, cat in enumerate(categories)}
df['label'] = df['Product'].map(label_map)
print("\nLabel Mapping:")
print(label_map)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply cleaning
df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)
print("✅ Text cleaning completed!")

# TF-IDF Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n🔹 {name} Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, target_names=categories))

# Best model (SVM)
best_model = LinearSVC().fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Test prediction
sample = ["I am receiving calls every day about a debt that is not mine."]
sample_tfidf = tfidf.transform(sample)
pred = best_model.predict(sample_tfidf)[0]
print("\n🔮 Sample Prediction:")
print("Text:", sample[0])
print("Predicted Category:", categories[pred])
