import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("UpdatedResumeDataSet.csv")  # Make sure file is in the same directory

# Simple text cleaning function
def clean_text(text_data):
    return text_data.lower()

df["Resume"] = df["Resume"].apply(clean_text)

# Select top 9 frequent job categories
top_9_categories = df["Category"].value_counts().nlargest(9).index.tolist()
df_top = df[df["Category"].isin(top_9_categories)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_top["Resume"], df_top["Category"], test_size=0.2, random_state=42
)

# Create pipeline with TF-IDF (using built-in 'english' stopwords) and linear SVM
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.8)),
    ("svm", SVC(kernel="linear", probability=True))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Print classification report
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer separately
joblib.dump(pipeline.named_steps["svm"], "svm_resume_classifier.pkl")
joblib.dump(pipeline.named_steps["tfidf"], "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved as 'svm_resume_classifier.pkl' and 'tfidf_vectorizer.pkl'")
