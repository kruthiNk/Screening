import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated dataset: Replace this with real labeled data
# Each entry represents domain keyword scores + label (target job role)
X = [
    [10, 5, 8, 7, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 5, 1],  # Junior Data Scientist
    [3, 2, 6, 5, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 6, 1],   # Data Analyst
    [2, 1, 1, 10, 10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 1],  # Software Engineer
    [0, 0, 0, 0, 0, 6, 10, 0, 0, 1, 0, 0, 0, 0, 3, 1],   # Web & Graphic Designer
    [0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 2, 1],   # Account Executive
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 10, 0, 0, 0, 0, 2, 1],   # Sales Representative
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 10, 0, 0, 3, 1],   # Content Creator
    [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 11, 0, 3, 1],   # Senior Accountant
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 3, 1],   # General Surgeon
]

y = [
    "Junior Data Scientist",
    "Data Analyst",
    "Software Engineer",
    "Web & Graphic Designer",
    "Account Executive",
    "Sales Representative",
    "Content Creator",
    "Senior Accountant",
    "General Surgeon"
]

# Train/test split and scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_scaled, y)

# Save model and scaler
with open('svm_model.pkl', 'wb') as f:
    pickle.dump((scaler, clf), f)

print("✅ SVM model trained and saved to 'svm_model.pkl'")
