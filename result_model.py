import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create sample student dataset
data = {
    "attendance": [90, 85, 70, 95, 60, 80, 75, 88, 92, 65,
                   78, 84, 91, 55, 68, 82, 96, 73, 87, 62],
    "internal_marks": [85, 80, 65, 90, 50, 75, 70, 82, 88, 55,
                       72, 78, 86, 45, 60, 77, 92, 68, 81, 52],
    "result": [1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
               1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["attendance", "internal_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
with open("student_result_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model and metrics saved successfully.")
