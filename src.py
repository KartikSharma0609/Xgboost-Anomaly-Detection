import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# =========================
# SMART UNDERSAMPLING
# =========================
def smart_undersample(X, y, ratio=10):
    df = X.copy()
    df["target"] = y

    minority = df[df.target == 1]
    majority = df[df.target == 0].sample(
        n=len(minority) * ratio,
        random_state=42
    )

    balanced = pd.concat([minority, majority])
    balanced = balanced.sample(frac=1, random_state=42)

    return balanced.drop(columns=["target"]), balanced["target"]

# =========================
# MAIN PIPELINE
# =========================
def train_test_and_submit():

    # =========================
    # 1. LOAD TRAIN DATA
    # =========================
    print("1. Loading training data...")
    try:
        df = pd.read_parquet("train.parquet")
    except:
        df = pd.read_csv("train.csv")

    print("Train shape:", df.shape)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=["target"])
    y = df["target"]

    # =========================
    # FEATURE HANDLING
    # =========================
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_datetime(X[col])
            except:
                X[col] = X[col].astype("category")

        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64") // 10**9

    y = LabelEncoder().fit_transform(y)

    # =========================
    # BALANCE DATA
    # =========================
    print("2. Balancing dataset...")
    X_bal, y_bal = smart_undersample(X, y, ratio=10)

    # =========================
    # SPLIT
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X_bal, y_bal,
        test_size=0.2,
        stratify=y_bal,
        random_state=42
    )

    # =========================
    # MODEL
    # =========================
    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=1.5,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("3. Training model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # =========================
    # THRESHOLD OPTIMIZATION
    # =========================
    print("4. Optimizing threshold...")
    probs = model.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_thresh = 0.5

    for t in np.arange(0.1, 0.9, 0.002):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("=" * 40)
    print("BEST F1:", round(best_f1, 4))
    print("BEST THRESHOLD:", round(best_thresh, 3))
    print("=" * 40)

    # =========================
    # RETRAIN ON FULL DATA
    # =========================
    print("5. Retraining on full dataset...")
    model.fit(X, y)

    # =========================
    # 6. LOAD TEST DATA
    # =========================
    print("6. Loading test data...")
    try:
        df_test = pd.read_parquet("test.parquet")
    except:
        df_test = pd.read_csv("test.csv")

    print("Test shape:", df_test.shape)

    if "ID" in df_test.columns:
        test_ids = df_test["ID"]
        X_test = df_test.drop(columns=["ID"])
    else:
        print("WARNING: No ID column found, using index")
        test_ids = df_test.index
        X_test = df_test.copy()

    # =========================
    # PROCESS TEST FEATURES
    # =========================
    for col in X_test.columns:
        if X_test[col].dtype == "object":
            try:
                X_test[col] = pd.to_datetime(X_test[col])
            except:
                X_test[col] = X_test[col].astype("category")

        if pd.api.types.is_datetime64_any_dtype(X_test[col]):
            X_test[col] = X_test[col].astype("int64") // 10**9

    # =========================
    # 7. PREDICT & SAVE CSV
    # =========================
    print("7. Generating predictions...")
    test_probs = model.predict_proba(X_test)[:, 1]
    final_preds = (test_probs >= best_thresh).astype(int)

    submission = pd.DataFrame({
        "ID": test_ids,
        "target": final_preds
    })

    output_file = "submission_file.csv"
    submission.to_csv(output_file, index=False)

    print("=" * 40)
    print(f"SUCCESS: Submission saved as {output_file}")
    print("=" * 40)
    print(submission.head())

if __name__ == "__main__":
    train_test_and_submit()
