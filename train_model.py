
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ------------------------------------------------
# 1. FILE LOCATIONS
# ------------------------------------------------

PROJECT_FOLDER = Path(__file__).parent

DATA_FILE = PROJECT_FOLDER / "data" / "shopsense_dataset.csv"

MODEL_FOLDER = PROJECT_FOLDER / "models"

REPORT_FOLDER = PROJECT_FOLDER / "reports"

MODEL_FILE = MODEL_FOLDER / "shopsense_model.joblib"


# ------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------

# Numerical features
NUMERIC_FEATURES = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "ProductRelated_Duration"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "Weekend"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Revenue"


# ------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------

if not DATA_FILE.exists():

    print("Dataset not found!")
    print("Please run python download_data.py first.")

    raise SystemExit(1)


data = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Total records:", len(data))


# ------------------------------------------------
# 4. SEPARATE INPUT AND OUTPUT
# ------------------------------------------------

X = data[FEATURES].copy()

y = (
    data[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })
)

# Remove rows with invalid target values
valid_rows = y.notna()

X = X.loc[valid_rows].copy()

y = y.loc[valid_rows].astype(int)

# Convert categorical features into strings
for column in CATEGORICAL_FEATURES:
    X[column] = X[column].astype(str)


# ------------------------------------------------
# 5. SPLIT THE DATASET
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------
# 6. DATA PREPROCESSING
# ------------------------------------------------

# Fill missing numerical values
numeric_preprocessing = SimpleImputer(
    strategy="median"
)

# Fill missing categorical values and encode them
categorical_preprocessing = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_preprocessing,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_preprocessing,
            CATEGORICAL_FEATURES
        )
    ]
)


# ------------------------------------------------
# 7. CREATE RANDOM FOREST MODEL
# ------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ------------------------------------------------
# 8. TRAIN THE MODEL
# ------------------------------------------------

print("\nTraining ShopSense model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ------------------------------------------------
# 9. TEST THE MODEL
# ------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------
# 10. EVALUATE PERFORMANCE
# ------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ------------------------------------------------
# 11. DISPLAY RESULTS
# ------------------------------------------------

print("\n----- ShopSense Model Results -----")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("Precision:", round(precision * 100, 2), "%")

print("Recall:", round(recall * 100, 2), "%")

print("F1 Score:", round(f1 * 100, 2), "%")

print("ROC-AUC:", round(roc_auc, 4))


# ------------------------------------------------
# 12. SAVE MODEL AND RESULTS
# ------------------------------------------------

MODEL_FOLDER.mkdir(exist_ok=True)

REPORT_FOLDER.mkdir(exist_ok=True)

# Save trained model
joblib.dump(model, MODEL_FILE)

# Save evaluation results
results = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4)
}

report_file = REPORT_FOLDER / "shopsense_results.json"

report_file.write_text(
    pd.Series(results).to_json(indent=4),
    encoding="utf-8"
)

print("\nModel saved successfully!")
print("Model location:", MODEL_FILE)

print("\nShopSense training completed!")
