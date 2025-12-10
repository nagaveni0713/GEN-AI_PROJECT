# 📊 ABC X1 Smartwatch — Sentiment Analyzer

A production-ready sentiment analysis system built to extract actionable insights from customer reviews of the **ABC X1 Smartwatch**. The solution uses **BERT Transformers** and is deployed as a **Flask Web Application** with a clean, professional UI.

---

## 🚀 Project Overview

This project implements a **state-of-the-art sentiment analysis pipeline** using the *nlptown/bert-base-multilingual-uncased-sentiment* model. It classifies customer reviews into **Positive**, **Neutral**, and **Negative** sentiments to support product development and customer experience teams.

---

## 🤖 Models Used

### **Phase 1 — Classical Machine Learning (Baseline Models)**

Trained and evaluated 6 ML models:

* Logistic Regression (with class balancing)
* Multinomial Naive Bayes
* Random Forest (100 trees)
* Decision Tree
* K-Nearest Neighbors
* Support Vector Machine (Linear Kernel)

**Result:** All models performed below 50% accuracy.
**Why:**

* Limited contextual understanding
* Cannot interpret nuances ("not bad" → positive)
* Sensitive to small and diverse text data

---

### **Phase 2 — Transformer Model (Final Model)**

**Model:** `nlptown/bert-base-multilingual-uncased-sentiment`

**Why BERT?**

* Pre-trained on millions of product reviews
* Understands context and subtle language cues
* Outputs star ratings (1–5) with high accuracy
* Requires *no additional training*

**Output Mapping:**

* **1–2 stars → Negative**
* **3 stars → Neutral**
* **4–5 stars → Positive**

**Benefits:**

* ✔ No manual feature engineering
* ✔ Handles nuanced text
* ✔ Consistently high accuracy
* ✔ Fast and reliable for deployment

---

## 📋 Workflow Summary

### **Milestone 1: Data Preparation**

* Dataset: `smart_watch_review.csv` (6000+ reviews)
* Removed duplicates & nulls
* Generated sentiment labels from star ratings
* (Initial phase only) Applied stemming, stopword removal, TF-IDF, and data augmentation

### **Milestone 2: Exploratory Data Analysis**

* Sentiment distribution
* Rating patterns
* Identified class imbalance

### **Milestone 3: Model Development**

* Tested 6 classical models → poor performance
* Adopted BERT → high accuracy
* Selected BERT for deployment

### **Milestone 4: Deployment**

* Built using **Flask**
* Includes professional HTML/CSS frontend
* Returns:

  * Sentiment (Pos/Neu/Neg)
  * Confidence score (%)
  * Polarity score (–1 to +1)

### **Milestone 5: Documentation & Testing**

* README created
* Edge-case testing performed
* Fixed common deployment issues (404, port conflicts)

---

## 🧪 Sample Performance (Manual Testing)

| Review                                | True     | Predicted | Result    |
| ------------------------------------- | -------- | --------- | --------- |
| “Worst watch ever. 1 star.”           | Negative | Negative  | ✅ Correct |
| “It is okay, average.”                | Neutral  | Neutral   | ✅ Correct |
| “Amazing watch, love it!”             | Positive | Positive  | ✅ Correct |
| “The watch keeps lagging…”            | Negative | Negative  | ✅ Correct |
| “The watch is alright for basic use.” | Neutral  | Neutral   | ✅ Correct |

> **Note:** Some noisy labels exist in the dataset, but manual testing shows BERT performs reliably.

---

## 🛠️ Technology Stack

* Python 3.8+
* Hugging Face Transformers (BERT)
* PyTorch
* Flask
* Pandas
* Scikit-learn
* HTML/CSS

---

## ▶️ How to Run the Project

### **Install Dependencies**

```bash
pip install transformers torch scipy flask pandas scikit-learn
```

### **Run the Flask App**

```bash
python app.py
```

### **Open the Web Interface**

```
http://127.0.0.1:5001
```

### **Usage**

1. Enter a customer review
2. Click **Analyze Sentiment**
3. View:

   * Sentiment (Positive / Neutral / Negative)
   * Confidence %
   * Polarity score (–1 to +1)

---

## 📂 Project Structure

```
genai/
├── app.py                      # Flask application with BERT model
├── Sentiment_Analysis.ipynb    # Notebook for exploration & evaluation
├── templates/
│   └── index.html              # Frontend UI
├── data/
│   └── smart_watch_review.csv  # Dataset
└── README.md                   # Project documentation
```

---

## 🎯 Key Learnings

* Classical ML struggles with nuanced sentiment interpretation
* Pre-trained BERT models offer superior performance with no training required
* Data quality significantly impacts evaluation metrics
* Flask is ideal for lightweight ML deployment

---

## 👨‍💻 Author

Project developed as part of the **ABC Company ML Initiative** to analyze feedback for the **ABC X1 Smartwatch**.
