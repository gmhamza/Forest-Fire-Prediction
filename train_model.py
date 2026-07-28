"""
train_model.py
---------------
Forest Fire Risk Prediction - Model Training Script

WHAT THIS SCRIPT DOES (step by step):
1. Loads the dataset (data/Algerian_forest_fires_dataset.csv)
2. Cleans it (missing values, duplicates)
3. Encodes categorical columns (Region, Classes)
4. Splits into train/test sets
5. Scales numeric features
6. Trains 6 different classification algorithms
7. Compares them using Accuracy, Precision, Recall, F1-score
8. Picks the BEST model automatically
9. Saves the best model + the scaler + the encoders as .pkl files
   (these .pkl files are what app.py loads to make live predictions)

Run this file first, before running the Streamlit app:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

print("=" * 60)
print("FOREST FIRE RISK PREDICTION - MODEL TRAINING")
print("=" * 60)

# -----------------------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------------------
# Fix escape sequence warning by using forward slashes or a raw string (r"...")
df = pd.read_csv("data/Algerian_forest_fires_dataset.csv")

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Add the missing 'Region' column based on dataset structure
# (Rows up to index 121 are Bejaia; index 122+ are Sidi Bel-abbes)
df.loc[:121, "Region"] = "Bejaia"
df.loc[122:, "Region"] = "Sidi Bel-abbes"

# Remove the secondary header row in the middle of the dataset and missing values
df = df.dropna().reset_index(drop=True)
df = df[df["day"] != "day"].reset_index(drop=True)
# -----------------------------------------------------------
# STEP 2: Data Cleaning
# -----------------------------------------------------------
# Why: Real-world data always has missing values / duplicate rows.
# ML models cannot handle NaN values, so we fill them.
# We remove duplicates so the model does not "memorize" repeated rows.
df_clean = df.drop_duplicates().copy()

# Strip whitespace from column names
df_clean.columns = df_clean.columns.str.strip()

numeric_cols = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

# Convert numeric columns to numeric type, coercing errors to NaN
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Fill missing values with the median after conversion
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

df_clean["Classes"] = df_clean["Classes"].str.strip().str.lower()

# -----------------------------------------------------------
# STEP 3: Encoding categorical columns
# -----------------------------------------------------------
# Why: ML algorithms only understand numbers, not text like
# "Bejaia" or "fire". LabelEncoder converts text categories to numbers.
region_encoder = LabelEncoder()
df_clean["Region_enc"] = region_encoder.fit_transform(df_clean["Region"])

target_encoder = LabelEncoder()
df_clean["Classes_enc"] = target_encoder.fit_transform(df_clean["Classes"])  # fire=0/1
print(f"[3] Encoded 'Region' -> {list(region_encoder.classes_)}")
print(f"[3] Encoded 'Classes' -> {list(target_encoder.classes_)}")

# -----------------------------------------------------------
# STEP 4: Feature Selection
# -----------------------------------------------------------
# Why: 'day', 'year' carry no predictive signal about fire risk.
# We keep weather + FWI-system features + region, which are the
# scientifically meaningful drivers of forest fire risk.
feature_cols = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC",
                 "ISI", "BUI", "FWI", "month", "Region_enc"]

X = df_clean[feature_cols]
y = df_clean["Classes_enc"]

# -----------------------------------------------------------
# STEP 5: Train-Test Split
# -----------------------------------------------------------
# Why: We train on 80% of the data and test on the remaining 20%
# that the model has NEVER seen, to fairly judge real performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[5] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------------------------------------
# STEP 6: Feature Scaling
# -----------------------------------------------------------
# Why: Features like DC (0-220) and Rain (0-16) are on very different
# scales. Algorithms like Logistic Regression, KNN and SVM are
# distance/gradient based and perform poorly without scaling.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[6] Applied StandardScaler to features.")

# -----------------------------------------------------------
# STEP 7: Train multiple models & compare
# -----------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
trained_models = {}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1 Score": f1
    })
    trained_models[name] = model

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds, target_names=target_encoder.classes_))

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string(index=False))

# -----------------------------------------------------------
# STEP 8: Select best model
# -----------------------------------------------------------
# Why F1-score (not just accuracy): our classes are fairly balanced
# here, but F1 is a safer general-purpose metric because it balances
# false alarms (precision) against missed fires (recall) - and in a
# fire warning system, missing a real fire (false negative) is costly.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\n[8] BEST MODEL SELECTED: {best_model_name}")

# -----------------------------------------------------------
# STEP 9: Save model, scaler, encoders using Pickle
# -----------------------------------------------------------
# Why Pickle: it saves the trained Python object exactly as-is to disk,
# so the Streamlit app can load it instantly without retraining.
with open("models/forest_fire_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/region_encoder.pkl", "wb") as f:
    pickle.dump(region_encoder, f)

with open("models/target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

results_df.to_csv("models/model_comparison_results.csv", index=False)

print("\n[9] Saved to /models:")
print("    - forest_fire_model.pkl  (best trained model)")
print("    - scaler.pkl             (StandardScaler)")
print("    - region_encoder.pkl     (LabelEncoder for Region)")
print("    - target_encoder.pkl     (LabelEncoder for Classes)")
print("    - feature_columns.pkl    (column order used for prediction)")
print("    - model_comparison_results.csv")
print("\nTraining complete. You can now run: streamlit run app.py")