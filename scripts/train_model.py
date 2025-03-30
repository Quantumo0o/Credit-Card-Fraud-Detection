import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Load dataset
import os
import pandas as pd

# Check if the local file exists
file_path = "creditcard.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Original Dataset Shape:", df.shape)
else:
    print(f"{file_path} not found. Loading dataset from KaggleHub...")
    
    # Install dependencies as needed:
    # pip install kagglehub[pandas-datasets]
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    # Load the latest version of the dataset
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mlg-ulb/creditcardfraud",
        file_path,
    )

    print("First 5 records:", df.head())

# 2️⃣ Remove duplicates
df_cleaned = df.drop_duplicates()
print("Dataset Shape After Cleaning:", df_cleaned.shape)

# 3️⃣ Separate features & target
X = df_cleaned.drop(columns=["Class"])
y = df_cleaned["Class"]

# 4️⃣ Normalize "Amount" and "Time"
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# 5️⃣ Hybrid Balancing (Under-sampling + SMOTE)
print("\n📌 Class distribution before balancing:", Counter(y))

# Step 1: Undersample the majority class
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_under, y_under = undersample.fit_resample(X, y)

# Step 2: Apply SMOTE oversampling
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

# Check new class distribution
print("📌 Class distribution after hybrid balancing:", Counter(y_balanced))

# 6️⃣ Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

# 7️⃣ Train XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

print("\n🔄 Training XGBoost model...")
xgb_model.fit(X_train, y_train)
print("✅ Model Training Complete!\n")

# 8️⃣ Predictions
y_pred = xgb_model.predict(X_test)

# 9️⃣ Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"📊 Accuracy: {accuracy:.6f}")
print(f"📊 Precision: {precision:.6f}")
print(f"📊 Recall: {recall:.6f}")
print(f"📊 F1 Score: {f1:.6f}")

print("\n📜 Classification Report:\n", classification_report(y_test, y_pred))

# 🔟 Save model and scaler
joblib.dump(xgb_model, "models/credit_card_fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\n✅ Model & Scaler saved successfully!")
