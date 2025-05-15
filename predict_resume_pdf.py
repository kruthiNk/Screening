import sys
import fitz  # PyMuPDF
import joblib

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def predict_category_from_pdf(pdf_path):
    # Load model and vectorizer
    svm_model = joblib.load("svm_resume_classifier.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_path)
    resume_text = resume_text.lower()

    # Vectorize
    vectorized_text = tfidf_vectorizer.transform([resume_text])

    # Predict
    prediction = svm_model.predict(vectorized_text)
    return prediction[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_resume_pdf.py <resume_pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    category = predict_category_from_pdf(pdf_file)
    print(f"Predicted Job Category: {category}")
