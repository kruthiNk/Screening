import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction import text
import joblib

# Load data
df = pd.read_csv("resumes.csv")  # Ensure the file has 'Resume' and 'Category'

# Clean text function
def clean_text(text_data):
    return text_data.lower()

df["Resume"] = df["Resume"].apply(clean_text)

# Get top 9 most frequent categories
top_9_categories = df["Category"].value_counts().nlargest(9).index.tolist()
df_top = df[df["Category"].isin(top_9_categories)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_top["Resume"], df_top["Category"], test_size=0.2, random_state=42
)

# TF-IDF + SVM pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, max_df=0.8)),
    ("svm", SVC(kernel="linear", probability=True))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(pipeline.named_steps["svm"], "svm_resume_classifier.pkl")
joblib.dump(pipeline.named_steps["tfidf"], "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved as 'svm_resume_classifier.pkl' and 'tfidf_vectorizer.pkl'")
