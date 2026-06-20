# 🏃 Human Activity Recognition System

> A machine learning system that classifies human physical activities from smartphone sensor data using multiple supervised learning algorithms and an interactive Streamlit dashboard.

---

## 📌 Overview

This project implements a **Human Activity Recognition (HAR)** system using the **UCI Human Activity Recognition Using Smartphones Dataset**. The system analyzes accelerometer and gyroscope sensor data collected from a waist-mounted smartphone and predicts the physical activity being performed.

Several machine learning algorithms were trained, tuned, and evaluated, including:

* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest
* Decision Tree

The best-performing model was selected based on test accuracy and deployed through a Streamlit web application.

---

## 🎯 Activities Classified

| Activity ID | Activity Name      |
| ----------- | ------------------ |
| 1           | WALKING            |
| 2           | WALKING_UPSTAIRS   |
| 3           | WALKING_DOWNSTAIRS |
| 4           | SITTING            |
| 5           | STANDING           |
| 6           | LAYING             |

---

## 📁 Project Structure

```text
Human_Activity_Recognition/
│
├── app.py
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── best_model.pkl
│   ├── rf_tuned.pkl
│   ├── svm_tuned.pkl
│   ├── lr_tuned.pkl
│   ├── dt_tuned.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── tuned_accuracy_comparison.png
│   ├── tuned_model_summary.txt
│   ├── best_model_confusion_matrix.png
│   └── best_model_classification_report.txt
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── train_tuned.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Used

| Model                        | Description                     |
| ---------------------------- | ------------------------------- |
| Support Vector Machine (SVM) | Margin-based classifier         |
| Logistic Regression          | Linear classification model     |
| Random Forest                | Ensemble of decision trees      |
| Decision Tree                | Tree-based classification model |

### Scaling Strategy

* SVM → StandardScaler applied
* Logistic Regression → StandardScaler applied
* Random Forest → Raw features used
* Decision Tree → Raw features used

---

## 📊 Dataset Information

Dataset: UCI Human Activity Recognition Using Smartphones

### Dataset Characteristics

* 30 volunteers
* Age range: 19–48 years
* Samsung Galaxy S II smartphone
* Accelerometer and Gyroscope sensors
* Sampling frequency: 50 Hz
* 561 engineered features
* 6 activity classes

### Features

The dataset contains:

* Time-domain features
* Frequency-domain features
* Signal magnitude features
* Statistical measurements

Examples:

* tBodyAcc-mean()-X
* tBodyAcc-std()-Y
* tBodyGyro-mad()-Z
* fBodyAcc-energy()-X

---

## ⚙️ Data Preprocessing

The preprocessing pipeline performs:

1. Dataset loading
2. Feature-label separation
3. Standard scaling (for SVM and Logistic Regression)
4. Train-test preparation
5. Feature transformation

---

## 🔧 Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used for model tuning.

### Random Forest Parameters

* n_estimators
* max_depth
* min_samples_split

### SVM Parameters

* C
* kernel
* gamma

### Logistic Regression Parameters

* C
* solver
* max_iter

### Decision Tree Parameters

* max_depth
* min_samples_split
* criterion

---

## 📈 Model Performance

| Model                  | Test Accuracy |
| ---------------------- | ------------- |
| Decision Tree          | 85.95%        |
| Random Forest          | 92.74%        |
| Logistic Regression    | 95.52%        |
| Support Vector Machine | 96.17%        |

### Best Model

**Support Vector Machine (SVM)**

* Test Accuracy: 96.17%
* Train Accuracy: 99.27%
* Overfitting Gap: 3.10%

The tuned SVM achieved the highest generalization performance on the test dataset.

---

## 📊 Evaluation Metrics

The system evaluates models using:

* Accuracy
* Confusion Matrix
* Classification Report
* Overfitting Gap Analysis
* Cross-Validation Performance

---

## 🚀 Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Baseline Models

```bash
python src/train.py
```

### Train Tuned Models

```bash
python src/train_tuned.py
```

### Evaluate Models

```bash
python src/evaluate.py
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

---

## 🖥️ Streamlit Dashboard Features

* Interactive prediction interface
* Real-time activity classification
* Model performance metrics
* Confusion matrix visualization
* Accuracy comparison charts
* Classification report display
* Dataset preview

---

## 📦 Requirements

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## 🎓 Academic Objectives

This project demonstrates:

* Data preprocessing
* Feature engineering
* Supervised machine learning
* Hyperparameter tuning
* Model evaluation
* Overfitting analysis
* Model deployment using Streamlit

---

## 👨‍💻 Author

**Kashaf Ejaz**

Bachelor of Science in Artificial Intelligence (BSAI)

Riphah International University, Lahore

---

## 📄 License

This project was developed for academic and educational purposes.

Dataset credit belongs to the creators of the UCI Human Activity Recognition Using Smartphones Dataset.
