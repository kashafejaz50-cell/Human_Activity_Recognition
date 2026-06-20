# src/evaluate.py

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.preprocess import load_data, preprocess


print("✅ Starting evaluation...\n")


# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

train_df, test_df = load_data()

print("✅ Dataset loaded")


# =====================================================
# STEP 2: PREPROCESS DATA
# =====================================================

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = preprocess(train_df, test_df)

print("✅ Preprocessing completed")


# =====================================================
# STEP 3: LOAD TUNED MODEL
# =====================================================

model = joblib.load("models/best_model.pkl")

print("✅ Best model loaded")


# =====================================================
# STEP 4: MAKE PREDICTIONS
# =====================================================

# Models requiring scaled data
if isinstance(model, (SVC, LogisticRegression)):

    y_pred = model.predict(X_test_scaled)

    print("📌 Using SCALED test data")

# Tree-based models use raw data
else:

    y_pred = model.predict(X_test)

    print("📌 Using RAW test data")

print("✅ Predictions completed")


# =====================================================
# STEP 5: ACCURACY
# =====================================================

acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print("📊 MODEL EVALUATION RESULTS")
print("==============================\n")

print(f"Accuracy: {acc:.4f}")


# =====================================================
# STEP 6: CLASSIFICATION REPORT
# =====================================================

report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")
print(report)


# =====================================================
# STEP 7: SAVE REPORT
# =====================================================

os.makedirs("reports", exist_ok=True)

report_path = "reports/best_model_classification_report.txt"

with open(report_path, "w") as f:

    f.write("HUMAN ACTIVITY RECOGNITION - BEST MODEL REPORT\n")
    f.write("=============================================\n\n")

    f.write(f"Accuracy: {acc:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

print(f"✅ Classification report saved at {report_path}")

# =====================================================
# STEP 8: CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - HAR (Best Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

cm_path = "reports/best_model_confusion_matrix.png"

plt.savefig(cm_path)

plt.close()

print(f"✅ Confusion matrix saved at {cm_path}")

# =====================================================
# STEP 9: FINISHED
# =====================================================

print("\n🎉 Evaluation completed successfully!")
print("📁 Check reports/ folder")