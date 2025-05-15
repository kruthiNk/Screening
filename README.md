# AI -based Resume Screening

This repository contains a machine learning pipeline to classify resumes into job categories based on their content. The project uses TF-IDF vectorization and a Support Vector Machine (SVM) classifier trained on a curated resume dataset. It supports prediction directly from PDF resume files.

---

## Repository Structure

| File Name                 | Description                                            |
|---------------------------|--------------------------------------------------------|
| `UpdatedResumeDataSet.csv`| Dataset containing resumes and their job categories.  |
| `classification_jobs.txt` | List of job categories used for classification.       |
| `train_svm.py`            | Script to train the TF-IDF + SVM model on the dataset.|
| `predict_resume_pdf.py`   | Script to predict job category from an input PDF resume.|
| `svm_resume_classifier.pkl` | Trained SVM model saved using joblib.                |
| `tfidf_vectorizer.pkl`    | Trained TF-IDF vectorizer saved using joblib.         |

---

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- joblib
- PyPDF2 (for PDF text extraction)
- Streamlit (optional, if you want to build an interactive UI)

Install dependencies with:

```bash
pip install pandas scikit-learn joblib PyPDF2 streamlit
