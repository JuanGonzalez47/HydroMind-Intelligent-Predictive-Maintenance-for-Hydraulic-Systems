import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

warnings.filterwarnings("ignore")

window_seconds = 45

# Frequencies of sensors
sensor_sampling_freq = {
    "CE": 1,
    "CP": 1,
    "EPS1": 100,
    "FS1": 10,
    "FS2": 10,
    "profile": 1,
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,
    "SE": 1,
    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,
    "VS1": 1
}

# Paths
BASE_DIR = Path.cwd()
synthetic_dir = BASE_DIR / "database" / "bronce"

files = [
    "CE.txt", "CP.txt", "EPS1.txt", "FS1.txt", "FS2.txt", "profile.txt",
    "PS1.txt", "PS2.txt", "PS3.txt", "PS4.txt", "PS5.txt", "PS6.txt",
    "SE.txt", "TS1.txt", "TS2.txt", "TS3.txt", "TS4.txt", "VS1.txt"
]

# upload data
print("\nuploading data...")
dataset = {}
for file in files:
    path = os.path.join(synthetic_dir, file)
    df = pd.read_csv(path, sep="\t", header=None)
    sensor_name = file.replace(".txt", "")
    if sensor_name != "profile":
        freq = sensor_sampling_freq.get(sensor_name, 1)
        n_window = window_seconds * freq
        if df.shape[1] > n_window:
            df = df.iloc[:, :n_window]
    dataset[sensor_name] = df
print("data uploaded.")

# slope function
def calc_slope(x):
    idx = np.arange(len(x))
    if len(x) < 2:
        return 0.0
    p = Polynomial.fit(idx, x, 1)
    coef = p.convert().coef
    if len(coef) < 2:
        return 0.0
    return coef[1]

# features extraction
stats_functions = {
    "mean": np.mean,
    "std": np.std,
    "skew": skew,
    "kurtosis": kurtosis,
    "slope": calc_slope,
    "max": np.max
}
features = []
print("\nextracting features...")
for sensor, df in dataset.items():
    if sensor == "profile":
        continue
    for stat_name, func in stats_functions.items():
        col_name = f"{sensor}_{stat_name}"
        stat_values = df.apply(func, axis=1)
        features.append(stat_values.rename(col_name))

X_features = pd.concat(features, axis=1)
print("features extracted.")

if X_features.isnull().any().any():
    print("\nImputing missing values...")
    for col in X_features.columns:
        if X_features[col].isnull().sum() > 0:
            mean = X_features[col].mean()
            std = X_features[col].std()
            n_missing = X_features[col].isnull().sum()
            random_values = np.random.normal(loc=mean, scale=std, size=n_missing)
            X_features.loc[X_features[col].isnull(), col] = random_values
    print("Missing values imputed.")

# targets
profile_columns = [
    "cooler_condition", "valve_condition", "internal_pump_leakage",
    "hydraulic_accumulator", "stable_flag"
]
profile_df = dataset["profile"].copy()
profile_df.columns = profile_columns
X_full = pd.concat([X_features, profile_df], axis=1)

# scale
print("\nscaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_scaled = pd.DataFrame(X_scaled, columns=X_features.columns)
print("features scaled.")

selected_features = {
    "cooler_condition": ["CE_mean"],
    "valve_condition": [
        'EPS1_kurtosis', 'FS1_mean', 'FS1_skew', 'FS1_max', 'PS1_kurtosis', 'PS1_slope',
        'PS2_std', 'PS2_skew', 'PS2_kurtosis', 'PS2_slope', 'SE_mean', 'SE_skew', 'SE_kurtosis', 'SE_slope'
    ],
    "internal_pump_leakage": [
        'EPS1_mean', 'EPS1_max', 'FS1_mean', 'FS1_std', 'FS1_slope', 'PS1_max', 'PS3_mean',
        'SE_mean', 'SE_std', 'SE_slope', 'SE_max'
    ],
    "hydraulic_accumulator": [
        'CE_mean', 'EPS1_mean', 'FS1_skew', 'FS1_kurtosis', 'FS2_mean', 'FS2_max', 'PS1_std',
        'PS3_std', 'PS3_skew', 'PS3_kurtosis', 'PS5_mean', 'PS5_max', 'PS6_mean', 'PS6_max',
        'SE_mean', 'SE_skew', 'SE_kurtosis', 'SE_max', 'TS1_mean', 'TS1_max', 'TS2_mean',
        'TS2_slope', 'TS2_max', 'TS3_mean', 'TS3_max', 'TS4_mean', 'TS4_max'
    ],
    "stable_flag": [
        'FS1_mean', 'FS1_std', 'FS1_skew', 'FS1_kurtosis', 'FS1_slope', 'PS1_skew', 'PS2_std',
        'PS2_skew', 'PS2_kurtosis', 'PS2_slope', 'PS5_mean', 'PS6_mean', 'SE_mean', 'SE_std',
        'SE_skew', 'SE_kurtosis', 'SE_max', 'TS4_mean', 'TS4_max'
    ]
}

# model configurations
model_configs = {
    "cooler_condition": {
        "model": DecisionTreeClassifier(max_depth=3, random_state=42),
        "features": selected_features["cooler_condition"],
        "label_encoder": False
    },
    "valve_condition": {
        "model": SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42),
        "features": selected_features["valve_condition"],
        "label_encoder": True
    },
    "internal_pump_leakage": {
        "model": SVC(kernel='rbf', C=50, gamma=0.01, class_weight='balanced', decision_function_shape='ovr', random_state=42),
        "features": selected_features["internal_pump_leakage"],
        "label_encoder": True
    },
    "hydraulic_accumulator": {
        "model": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features='log2', class_weight='balanced_subsample', random_state=42),
        "features": selected_features["hydraulic_accumulator"],
        "label_encoder": True
    },
    "stable_flag": {
        "model": RandomForestClassifier(n_estimators=300, max_depth=None, max_features='sqrt', random_state=42),
        "features": selected_features["stable_flag"],
        "label_encoder": True
    }
}

# results
results = []

for target, config in model_configs.items():
    print(f"\nTraining and evaluating: {target}")
    X = X_scaled[config["features"]]
    y = X_full[target]
    # code for label encoding, splitting, training, predicting, evaluating, and printing results goes here (same as in test_models.py)
    if config["label_encoder"]:
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    model = config["model"]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # decode predictions for evaluation
    if config["label_encoder"]:
        y_test_eval = le.inverse_transform(y_test)
        y_pred_eval = le.inverse_transform(y_pred)
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred
    # Metrics
    accuracy = accuracy_score(y_test_eval, y_pred_eval)
    precision = precision_score(y_test_eval, y_pred_eval, average="weighted", zero_division=0)
    recall = recall_score(y_test_eval, y_pred_eval, average="weighted", zero_division=0)
    f1 = f1_score(y_test_eval, y_pred_eval, average="weighted", zero_division=0)
    results.append({
        "target": target,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })
    # Confusion matrix and classification report
    print("\nconfusion matrix:")
    print(confusion_matrix(y_test_eval, y_pred_eval))
    print("\nclassification report:")
    print(classification_report(y_test_eval, y_pred_eval, zero_division=0))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1_score", ascending=False)
print("\nSummary Table:")
print(results_df)
