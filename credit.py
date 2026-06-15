import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# DATA CLEANING
# ==========================================

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

df = pd.get_dummies(df, drop_first=True)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)

# ==========================================
# SAVE RESULTS
# ==========================================

with open("results.txt", "w") as f:
    f.write("CUSTOMER CHURN PREDICTION RESULTS\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.close()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.close()

print("\nFiles Generated Successfully!")
print("results.txt")
print("confusion_matrix.png")
print("feature_importance.png")