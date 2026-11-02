# ⚙️ Machine Failure Prediction System

An end-to-end Machine Learning project that predicts potential machine failures using sensor readings and a Random Forest classification model.

## 📌 Project Overview

Unexpected machine failures can cause production downtime and maintenance costs. This project demonstrates a predictive maintenance approach using machine sensor data to identify machines that may be at risk of failure.

The application allows users to enter sensor readings and receive a predicted failure probability through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze machine sensor data.
- Perform exploratory data analysis (EDA).
- Engineer meaningful features.
- Train and evaluate classification models.
- Handle class imbalance.
- Tune the classification threshold.
- Build an interactive prediction dashboard.

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

- 10,000 records
- 14 original columns
- 339 machine failure cases
- Synthetic predictive maintenance dataset

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 🤖 Machine Learning Workflow

1. Data collection and exploration
2. Data cleaning and preprocessing
3. Feature engineering
4. Train-test splitting
5. Logistic Regression
6. Random Forest Classification
7. Model evaluation
8. Threshold optimization
9. Model interpretation
10. Streamlit deployment interface

## 📈 Model Performance

Final Random Forest model evaluated on the held-out test set using a classification threshold of 0.20.

| Metric | Score |
|---|---:|
| Accuracy | 96.4% |
| Precision | 48.4% |
| Recall | 89.7% |
| F1-score | 63.0% |

The selected threshold prioritizes detecting more potential failures, while accepting some false alarms.

## ⚙️ Engineered Features

- Temperature Difference
- Mechanical Power (kW)

## 🖥️ Application Features

- Interactive sensor input fields
- Automatic feature calculation
- Failure probability estimation
- Threshold-based failure alerts
- Machine failure prediction dashboard

## 📁 Project Structure

```text
MACHINE_FAILURE_PREDICTION/
│
├── DATA/
│   └── ai4i2020.csv
│
├── models/
│   ├── machine_failure_model.pkl
│   └── failure_threshold.pkl
│
├── notebooks/
│   └── machine_failure_analysis.ipynb
│
├── screenshots/
├── docs/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd YOUR_REPOSITORY_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## ⚠️ Limitations

- The AI4I 2020 dataset is synthetic and does not represent live industrial sensor data.
- Model predictions are not guarantees of actual machine condition.
- Further validation on real-world industrial data is necessary before operational use.

## 👨‍💻 Author

**Chirag Bhageriya**

Machine Learning | Python | Predictive Maintenance