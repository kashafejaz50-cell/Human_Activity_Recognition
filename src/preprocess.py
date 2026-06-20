## src/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ----------------------------
# LOAD DATASETS
# ----------------------------
def load_data():

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    return train_df, test_df


# ----------------------------
# SPLIT FEATURES AND LABELS
# ----------------------------
def split_features_labels(df):

    # HAR dataset:
    # Last column = activity label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Remove subject column if exists
    if "subject" in X.columns:
        X = X.drop("subject", axis=1)

    return X, y


# ----------------------------
# SCALE FEATURES
# ----------------------------
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# ----------------------------
# MAIN PREPROCESS PIPELINE
# ----------------------------
def preprocess(train_df, test_df):

    # Split features and labels
    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )

    # Return BOTH raw and scaled data
    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ----------------------------
# TEST PREPROCESSING
# ----------------------------
if __name__ == "__main__":

    print("✅ PREPROCESSING STARTED\n")

    # Load dataset
    train_df, test_df = load_data()

    print("✅ Dataset loaded")

    # Run preprocessing
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = preprocess(train_df, test_df)

    print("✅ Features and labels separated")
    print("✅ Scaling completed\n")

    # Shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\n✅ Preprocessing completed successfully!")