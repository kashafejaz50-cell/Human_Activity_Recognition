# src/train.py

import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# Import preprocessing functions
from src.preprocess import load_data, preprocess


# ----------------------------
# STEP 1: LOAD DATASET
# ----------------------------
print("\n✅ Loading dataset...")

train_df, test_df = load_data()

print("✅ Dataset loaded successfully!")


# ----------------------------
# STEP 2: PREPROCESS DATA
# ----------------------------
print("\n✅ Preprocessing dataset...")

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = preprocess(train_df, test_df)

print("✅ Preprocessing completed!")


# ----------------------------
# STEP 3: DEFINE MODELS
# ----------------------------
models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}


# ----------------------------
# STEP 4: TRAIN + EVALUATE
# ----------------------------
results = {}
trained_models = {}

print("\n==============================")
print("🔵 TRAINING MODELS STARTED")
print("==============================\n")

for name, model in models.items():

    # IMPORTANT:
    # SVM + Logistic Regression require scaled data
    if name in ["SVM", "Logistic Regression"]:

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

    else:
        # Tree models can use raw data
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    results[name] = acc
    trained_models[name] = model

    print(f"{name} Accuracy: {acc:.4f}")


# ----------------------------
# STEP 5: SELECT BEST MODEL
# ----------------------------
best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

print("\n==============================")
print("🏆 BEST MODEL SELECTED")
print("==============================")

print("Model:", best_model_name)
print("Accuracy:", results[best_model_name])


# ----------------------------
# STEP 6: SAVE MODEL + SCALER
# ----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/baseline_best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Model saved successfully!")
print("✅ Scaler saved successfully!")
print("📁 Saved inside models/")


# ----------------------------
# STEP 7: SAVE BEST MODEL INFO
# ----------------------------
with open("reports/baseline_model_performance_summary.txt", "w") as f:

    f.write("HUMAN ACTIVITY RECOGNITION MODEL RESULTS\n")
    f.write("========================================\n\n")

    for model_name, score in results.items():
        f.write(f"{model_name}: {score:.4f}\n")

    f.write("\n")
    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write(f"BEST ACCURACY: {results[best_model_name]:.4f}\n")

print("✅ Performance summary saved!")

# ----------------------------
# STEP 7: ACCURACY COMPARISON PLOT
# ----------------------------

import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.bar(results.keys(), results.values())

plt.title("Model Accuracy Comparison (HAR Dataset)")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.xticks(rotation=30)

plt.savefig("reports/baseline_accuracy-comparison.png")

plt.close()

print("📊 Accuracy comparison chart saved!")