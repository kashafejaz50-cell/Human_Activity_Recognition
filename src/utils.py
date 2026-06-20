# src/utils.py

import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# =====================================================
# LABEL DECODER
# =====================================================

def decode_activity(label):

    activity_map = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }

    return activity_map.get(label, "UNKNOWN")


# =====================================================
# MODEL ACCURACY FORMATTER
# =====================================================

def format_accuracy(results_dict):

    print("\n==============================")
    print("📊 MODEL ACCURACY SUMMARY")
    print("==============================\n")

    for model, acc in results_dict.items():
        print(f"{model}: {acc:.4f}")


# =====================================================
# PREPARE INPUT
# =====================================================

def prepare_input(sample, scaler, model):

    sample = np.array(sample).reshape(1, -1)

    # Models requiring scaling
    if isinstance(model, (SVC, LogisticRegression)):

        return scaler.transform(sample)

    # Tree models use raw data
    return sample