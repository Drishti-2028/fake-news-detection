import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv", encoding="utf-16")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# Features and labels
X = data["text"]
y = data["label"]

# Vectorize
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ✅ Multiple models
models = {
    "Naive Bayes":          MultinomialNB(),
    "Logistic Regression":  LogisticRegression(max_iter=1000),
    "Decision Tree":        DecisionTreeClassifier(),
    "Random Forest":        RandomForestClassifier(n_estimators=100),
    "SVM":                  LinearSVC()
}

print("\n===== Model Comparison =====")
best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:25s} → Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} with {best_accuracy*100:.2f}% accuracy")

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Best model saved!")