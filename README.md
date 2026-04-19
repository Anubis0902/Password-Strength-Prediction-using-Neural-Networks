# 🔐 Password Strength Prediction using ANN

## 📌 Project Overview

This project predicts the **strength of a password** (Weak, Medium, Strong) using an **Artificial Neural Network (ANN)**.

Unlike traditional rule-based systems, this model uses **feature engineering + neural networks** to capture complex patterns in passwords.

---

## 🚀 Features

* Converts raw passwords into meaningful numerical features
* Uses **entropy and character distribution** for better accuracy
* Trains an **ANN model** to classify password strength
* Provides real-time prediction for user input

---

## 🧠 Approach

### 1. Feature Engineering

Each password is converted into:

* Length
* Uppercase count
* Lowercase count
* Digit count
* Special character count
* Entropy (randomness measure)

---

### 2. Data Preprocessing

* Removed raw password column
* Encoded target labels (Weak, Medium, Strong)
* Applied feature scaling using StandardScaler

---

### 3. Model (ANN)

* Input Layer: 6 features
* Hidden Layers: 64 → 32 neurons
* Output Layer: 3 classes (Softmax)
* Activation: ReLU
* Loss: Sparse Categorical Crossentropy

---

### 4. Training

* Train-Test Split: 80-20
* Optimizer: Adam
* Epochs: 15

---

## 📊 Evaluation

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 🧪 Example Predictions

| Password     | Prediction |
| ------------ | ---------- |
| weakpass     | Weak       |
| P@ss123      | Medium     |
| Xy@9$Lm#2024 | Strong     |

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## ▶️ How to Run

1. Clone repository

```bash
git clone <your-repo-link>
cd password-strength-ann
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook or script

```bash
python main.py
```

---

## 🎯 Future Improvements

* Add LSTM for sequence-based learning
* Deploy using Streamlit (web app)
* Improve feature engineering with keyboard patterns

---

## 🧠 Key Learning

This project demonstrates how **feature engineering + ANN** can outperform simple rule-based password strength systems.

---

## 👨‍💻 Author

Atharv Patil