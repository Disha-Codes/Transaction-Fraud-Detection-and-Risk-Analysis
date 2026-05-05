# ---------------- IMPORTS ----------------
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

from imblearn.over_sampling import SMOTE

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "../models")
DATASET_DIR = os.path.join(BASE_DIR, "../dataset")

# Create folders if not exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
data_path = os.path.join(DATASET_DIR, "cleaned_fraud_data.csv")
df = pd.read_csv(data_path)

# ---------------- PREPARATION ----------------
df['Fraud_Label'] = df['Fraud_Label'].astype(int)

# Remove leakage columns
X = df.drop([
    'Fraud_Label',
    'Timestamp',
    'User_ID',
    'Transaction_ID',
    'Risk_Score',
    'Failed_Transaction_Count_7d'
], axis=1)

y = df['Fraud_Label']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

print("Feature shape:", X.shape)

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SMOTE ----------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", X_train.shape)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=18,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.35
y_pred = (y_prob >= threshold).astype(int)

print("\nThreshold used:", threshold)

# ---------------- EVALUATION ----------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ---------------- ROC CURVE ----------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Save ROC image
roc_path = os.path.join("roc_curve.png")
plt.savefig(roc_path)
plt.close()

print(f"\nROC curve saved at: {roc_path}")

# ---------------- FEATURE IMPORTANCE ----------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop Features:\n")
print(importance.head(10))

# ---------------- SAVE MODEL ----------------
model_path = os.path.join(MODELS_DIR, "fraud_model.pkl")
features_path = os.path.join(MODELS_DIR, "model_features.pkl")

joblib.dump(model, model_path)
joblib.dump(X.columns, features_path)

print(f"\nModel saved at: {model_path}")
print(f"Feature file saved at: {features_path}")

# ---------------- PREDICTION FUNCTION ----------------
def predict_fraud(input_df):
    import joblib
    import pandas as pd
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "../models")

    # Load model & features
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    model_features = joblib.load(os.path.join(MODELS_DIR, "model_features.pkl"))

    # Convert input to dataframe
    input_df = pd.DataFrame(input_df)

    # Apply encoding
    input_df = pd.get_dummies(input_df)

    # Align columns
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Predict probability
    prob = model.predict_proba(input_df)[:, 1]

    threshold = 0.35
    prediction = (prob >= threshold).astype(int)

    return prediction, prob

# ---------------- TEST FUNCTION ----------------
sample = X_test.iloc[0:1]

pred, prob = predict_fraud(sample)

print("\nSample Prediction:")
print("Fraud Probability:", prob)
print("Prediction (1=Fraud, 0=Normal):", pred)

# ---------------- SAVE DASHBOARD CSV ----------------
X_test_copy = X_test.copy()
X_test_copy['Actual_Fraud'] = y_test.values
X_test_copy['Predicted_Fraud'] = y_pred
X_test_copy['Fraud_Probability'] = y_prob

csv_path = os.path.join(DATASET_DIR, "fraud_dashboard_data.csv")
X_test_copy.to_csv(csv_path, index=False)

print(f"\nDashboard data saved at: {csv_path}")