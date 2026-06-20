# src/train_tuned.py

import os
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.preprocess import load_data, preprocess


# =====================================================
# STEP 1: LOAD DATASET
# =====================================================
print("\n✅ Loading dataset...")

train_df, test_df = load_data()

print("✅ Dataset loaded successfully!")


# =====================================================
# STEP 2: PREPROCESS DATA
# =====================================================
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


# =====================================================
# STEP 3: PARAMETER GRIDS
# =====================================================

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"]
}

lr_params = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs"],
    "max_iter": [1000]
}

dt_params = {
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "criterion": ["gini", "entropy"]
}


# =====================================================
# STEP 4: STORAGE
# =====================================================
results = {}
trained_models = {}


# =====================================================
# STEP 5: RANDOM FOREST
# =====================================================
print("\n🌲 TUNING RANDOM FOREST")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

# Predictions
rf_train_pred = best_rf.predict(X_train)
rf_test_pred = best_rf.predict(X_test)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

results["Random Forest"] = rf_test_acc
trained_models["Random Forest"] = best_rf

print("Best Params:", rf_grid.best_params_)
print("Train Accuracy:", rf_train_acc)
print("Test Accuracy:", rf_test_acc)
print("Overfitting Gap:", rf_train_acc - rf_test_acc)


# =====================================================
# STEP 6: SVM
# =====================================================
print("\n🧠 TUNING SVM")

svm_grid = GridSearchCV(
    SVC(),
    svm_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

svm_grid.fit(X_train_scaled, y_train)

best_svm = svm_grid.best_estimator_

svm_train_pred = best_svm.predict(X_train_scaled)
svm_test_pred = best_svm.predict(X_test_scaled)

svm_train_acc = accuracy_score(y_train, svm_train_pred)
svm_test_acc = accuracy_score(y_test, svm_test_pred)

results["SVM"] = svm_test_acc
trained_models["SVM"] = best_svm

print("Best Params:", svm_grid.best_params_)
print("Train Accuracy:", svm_train_acc)
print("Test Accuracy:", svm_test_acc)
print("Overfitting Gap:", svm_train_acc - svm_test_acc)


# =====================================================
# STEP 7: LOGISTIC REGRESSION
# =====================================================
print("\n📈 TUNING LOGISTIC REGRESSION")

lr_grid = GridSearchCV(
    LogisticRegression(),
    lr_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

lr_grid.fit(X_train_scaled, y_train)

best_lr = lr_grid.best_estimator_

lr_train_pred = best_lr.predict(X_train_scaled)
lr_test_pred = best_lr.predict(X_test_scaled)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

results["Logistic Regression"] = lr_test_acc
trained_models["Logistic Regression"] = best_lr

print("Best Params:", lr_grid.best_params_)
print("Train Accuracy:", lr_train_acc)
print("Test Accuracy:", lr_test_acc)
print("Overfitting Gap:", lr_train_acc - lr_test_acc)


# =====================================================
# STEP 8: DECISION TREE
# =====================================================
print("\n🌳 TUNING DECISION TREE")

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_

dt_train_pred = best_dt.predict(X_train)
dt_test_pred = best_dt.predict(X_test)

dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

results["Decision Tree"] = dt_test_acc
trained_models["Decision Tree"] = best_dt

print("Best Params:", dt_grid.best_params_)
print("Train Accuracy:", dt_train_acc)
print("Test Accuracy:", dt_test_acc)
print("Overfitting Gap:", dt_train_acc - dt_test_acc)


# =====================================================
# STEP 9: SELECT BEST MODEL
# =====================================================
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n🏆 BEST MODEL SELECTED")
print("Model:", best_model_name)
print("Test Accuracy:", results[best_model_name])


# =====================================================
# STEP 10: SAVE ALL MODELS + BEST MODEL
# =====================================================

os.makedirs("models", exist_ok=True)

# Save individual tuned models (IMPORTANT for comparison)
joblib.dump(best_rf, "models/rf_tuned.pkl")
joblib.dump(best_svm, "models/svm_tuned.pkl")
joblib.dump(best_lr, "models/lr_tuned.pkl")
joblib.dump(best_dt, "models/dt_tuned.pkl")

# Save best model separately (for prediction/GUI)
joblib.dump(best_model, "models/best_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")


# =====================================================
# STEP 11: REPORT
# =====================================================
os.makedirs("reports", exist_ok=True)

with open("reports/tuned_model_summary.txt", "w") as f:
    f.write("TUNED MODEL RESULTS (WITH OVERFITTING CHECK)\n")
    f.write("==========================================\n\n")

    f.write(f"Random Forest Test: {rf_test_acc:.4f}\n")
    f.write(f"SVM Test: {svm_test_acc:.4f}\n")
    f.write(f"Logistic Regression Test: {lr_test_acc:.4f}\n")
    f.write(f"Decision Tree Test: {dt_test_acc:.4f}\n\n")

    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write(f"BEST ACCURACY: {results[best_model_name]:.4f}\n")


# =====================================================
# STEP 12: PLOT
# =====================================================
plt.figure(figsize=(8, 5))

plt.bar(results.keys(), results.values())

plt.title("Tuned Model Comparison")
plt.xlabel("Models")
plt.ylabel("Test Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("reports/tuned_accuracy_comparison.png")
plt.close()


print("\n🎉 TRAINING + TUNING COMPLETED SUCCESSFULLY!")