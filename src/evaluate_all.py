# src/evaluate_all.py

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocess import load_data, preprocess


# =====================================================
# STEP 1: LOAD DATA
# =====================================================

print("\n==============================")
print("📊 MODEL COMPARISON EVALUATION")
print("==============================\n")

train_df, test_df = load_data()

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = preprocess(train_df, test_df)

print("✅ Dataset loaded and preprocessed")


# =====================================================
# STEP 2: LOAD TRAINED MODELS
# =====================================================

models = {
    "Random Forest": joblib.load("models/rf_tuned.pkl"),
    "SVM": joblib.load("models/svm_tuned.pkl"),
    "Logistic Regression": joblib.load("models/lr_tuned.pkl"),
    "Decision Tree": joblib.load("models/dt_tuned.pkl")
}

print("✅ All models loaded")


# =====================================================
# STEP 3: STORE RESULTS
# =====================================================

results = {}

print("\n==============================")
print("🚀 Evaluating All Models")
print("==============================\n")


# =====================================================
# STEP 4: EVALUATION LOOP
# =====================================================

for name, model in models.items():

    print(f"\n🔵 Evaluating {name}...")

    # Use scaled data for linear models
    if name in ["SVM", "Logistic Regression"]:

        y_pred_test = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)

    # Tree-based models use raw data
    else:

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

    # ==============================
    # METRICS
    # ==============================

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    precision = precision_score(y_test, y_pred_test, average="weighted")
    recall = recall_score(y_test, y_pred_test, average="weighted")
    f1 = f1_score(y_test, y_pred_test, average="weighted")

    # Store results
    results[name] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Print results
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")


# =====================================================
# STEP 5: SUMMARY TABLE
# =====================================================

print("\n==============================")
print("📌 FINAL MODEL COMPARISON TABLE")
print("==============================\n")

for model, scores in results.items():

    print(f"{model}")
    print(f"  Train Accuracy : {scores['train_accuracy']:.4f}")
    print(f"  Test Accuracy  : {scores['test_accuracy']:.4f}")
    print(f"  Precision      : {scores['precision']:.4f}")
    print(f"  Recall         : {scores['recall']:.4f}")
    print(f"  F1 Score       : {scores['f1_score']:.4f}\n")


# =====================================================
# STEP 6: BEST MODEL SELECTION (FIXED)
# =====================================================

best_model = max(results, key=lambda x: results[x]["f1_score"])

print("\n==============================")
print("🏆 BEST MODEL RESULT")
print("==============================")

print(f"Best Model: {best_model}")
print(f"Test Accuracy: {results[best_model]['test_accuracy']:.4f}")
print(f"F1 Score: {results[best_model]['f1_score']:.4f}")


# =====================================================
# STEP 7: FINAL SUMMARY
# =====================================================

print("\n==============================")
print("📌 FINAL MODEL SUMMARY")
print("==============================")

for model, scores in results.items():

    print(f"{model}")
    print(f"  Train Accuracy: {scores['train_accuracy']:.4f}")
    print(f"  Test Accuracy : {scores['test_accuracy']:.4f}")
    print(f"  Precision     : {scores['precision']:.4f}")
    print(f"  Recall        : {scores['recall']:.4f}")
    print(f"  F1 Score      : {scores['f1_score']:.4f}\n")


print(f"🏆 FINAL BEST MODEL: {best_model}")
print(f"🎯 BEST F1 SCORE: {results[best_model]['f1_score']:.4f}")

print("\n🎉 MODEL COMPARISON COMPLETED SUCCESSFULLY!")