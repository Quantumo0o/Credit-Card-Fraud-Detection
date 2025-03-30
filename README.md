# 🚀 **Credit Card Fraud Detection Using XGBoost**

## 📌 **Overview**
This project focuses on detecting fraudulent credit card transactions using **XGBoost**. Given the **highly imbalanced dataset**, we applied **hybrid balancing techniques** to improve recall while minimizing false positives.

✅ **Key Highlights:**  
- **High Accuracy** (99.96%) while reducing false alarms.  
- **Hybrid Sampling** (SMOTE + Undersampling) for better class balance.  
- **XGBoost Model** chosen for its speed and performance.  
- **Scalable and Ready for Deployment**.  

---

## 📂 **Project Structure**
```
📂 Credit-Card-Fraud-Detection
│── 📂 data
│   ├── creditcard.csv         # Dataset (If too large, provide a link)
│── 📂 models
│   ├── credit_card_fraud_model.pkl  # Saved trained model
│   ├── scaler.pkl  # Saved scaler for preprocessing
│── 📂 scripts
│   ├── train_model.py  # Full training pipeline
│   ├── predict.py  # Script to test new transactions
│── 📜 README.md  # Documentation
│── 📜 requirements.txt  # Dependencies
│── 📜 .gitignore  # Exclude unnecessary files
```

---

## 🔧 **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Quantumo0o/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### **2️⃣ Install Dependencies**
Ensure you have Python **3.8+** installed. Then, run:
```bash
pip install -r requirements.txt
```

---

## 🚀 **Usage Guide**
### **1️⃣ Train the Model**
```bash
python scripts/train_model.py
```
🔹 This script will:  
✔ Load the dataset  
✔ Perform **data preprocessing**  
✔ Apply **hybrid balancing**  
✔ Train the **XGBoost model**  
✔ Save the **trained model & scaler**  

---

### **2️⃣ Predict a New Transaction**
```bash
python scripts/predict.py
```
This script will:  
✔ Load the **trained model**  
✔ Take a **new transaction** as input  
✔ Normalize it using the **saved scaler**  
✔ Predict whether it's **fraudulent or not**  

---

## 📊 **Model Performance**
| Metric      | Score  |
|------------|--------|
| **Accuracy**   | 99.96% |
| **Precision**  | 98.71% |
| **Recall**     | 78.57% |
| **F1-Score**   | 87.50% |

✅ **Why This Matters:**  
- **High accuracy** ensures correct predictions.  
- **High precision** minimizes false positives (avoids blocking real users).  
- **Balanced recall** ensures fraud detection isn't ignored.  

---

## 📈 **Exploratory Data Analysis (EDA)**
Key insights from the dataset:  
- **Fraudulent transactions are extremely rare** (~0.17% of the dataset).  
- **Fraudulent transactions tend to have higher variance** in amount.  
- **Fraud does not show a strong time-based pattern**, making feature engineering challenging.  

📌 **EDA and visualizations were performed in the scripts directly (not in notebooks).**  

---

## 🛠 **Technical Details**
### **1️⃣ Data Preprocessing**
✔ Removed **duplicate transactions**  
✔ Standardized **Time & Amount** using **StandardScaler**  
✔ Used **SMOTE + Undersampling** to handle class imbalance  

### **2️⃣ Model Selection**
We compared different models and found:  
- **Random Forest:** High accuracy but slower & overfits  
- **Logistic Regression:** Poor recall on fraud cases  
- **XGBoost (Best Choice):** Fast, high recall, & handles imbalance well  

---

## 🤖 **Next Steps**
✅ Deploy as a **Flask API** for real-time fraud detection  
✅ Optimize hyperparameters using **GridSearchCV**  
✅ Implement **anomaly detection** for unknown fraud patterns  

---

## 📜 **License**
This project is **open-source** under the **MIT License**.  

📢 **Contributions Welcome!** Feel free to submit issues or pull requests.  

---

🔥 **Enjoyed this project?** Give it a ⭐ on GitHub! 🚀  

---

## 🔗 **References**
- Dataset: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- XGBoost Docs: [Official XGBoost Guide](https://xgboost.readthedocs.io/en/stable/)  

---

This version **removes any mention of Jupyter notebooks** while keeping everything **detailed, structured, and beginner-friendly**. Let me know if you need any more refinements! 🚀
