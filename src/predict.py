# src/predict.py

import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.preprocess import load_data, preprocess


# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

train_df, test_df = load_data()


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


# =====================================================
# STEP 3: LOAD TUNED MODEL
# =====================================================

model = joblib.load("models/best_model.pkl")

print("✅ Best model loaded")


# =====================================================
# STEP 4: ACTIVITY LABELS
# =====================================================

activity_labels = {
    "WALKING": "WALKING",
    "WALKING_UPSTAIRS": "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS": "WALKING_DOWNSTAIRS",
    "SITTING": "SITTING",
    "STANDING": "STANDING",
    "LAYING": "LAYING"
}


# =====================================================
# STEP 5: USER INPUT
# =====================================================

print("\n==============================")
print("🔮 HUMAN ACTIVITY PREDICTION")
print("==============================\n")

row_index = int(input(f"Enter row index (0 to {len(X_test)-1}): "))


# =====================================================
# STEP 6: GET SAMPLE
# =====================================================

sample = X_test.iloc[[row_index]]

# Scaled version
sample_scaled = scaler.transform(sample)


# =====================================================
# STEP 7: PREDICT
# =====================================================

# Models requiring scaled input
if isinstance(model, (SVC, LogisticRegression)):

    prediction = model.predict(sample_scaled)[0]

    print("\n📌 Using SCALED sample")

# Tree models use raw data
else:

    prediction = model.predict(sample)[0]

    print("\n📌 Using RAW sample")


# =====================================================
# STEP 8: OUTPUT RESULT
# =====================================================

print("\n==============================")
print("📊 PREDICTION RESULT")
print("==============================\n")

actual_activity = y_test.iloc[row_index]

print("Row Index:", row_index)
print("Predicted Activity :", prediction)
print("Actual Activity    :", actual_activity)


# =====================================================
# STEP 9: CHECK CORRECTNESS
# =====================================================

if prediction == actual_activity:

    print("\n✅ Prediction Correct")

else:

    print("\n❌ Prediction Incorrect")