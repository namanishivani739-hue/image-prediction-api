"""
Trains a handwritten digit classifier on sklearn's built-in digits dataset
(no external download needed - dataset ships with scikit-learn).
Saves the trained model as model.pkl
"""
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (1797 images of handwritten digits, 8x8 pixels each)
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", gamma=0.001, C=10, probability=True)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc*100:.2f}%")

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
