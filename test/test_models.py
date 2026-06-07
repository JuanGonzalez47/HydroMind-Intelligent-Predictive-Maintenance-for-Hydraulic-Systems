import os
import joblib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from numpy.polynomial import Polynomial

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


warnings.filterwarnings("ignore")

window_seconds = 2

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

# paths

BASE_DIR = Path.cwd()

synthetic_dir = BASE_DIR / "database" / "synthetic"

models_dir = BASE_DIR / "models"

# sensor files

files = [
    "CE.txt",
    "CP.txt",
    "EPS1.txt",
    "FS1.txt",
    "FS2.txt",
    "profile.txt",
    "PS1.txt",
    "PS2.txt",
    "PS3.txt",
    "PS4.txt",
    "PS5.txt",
    "PS6.txt",
    "SE.txt",
    "TS1.txt",
    "TS2.txt",
    "TS3.txt",
    "TS4.txt",
    "VS1.txt"
]


# load synthetic raw data

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

print("\nsynthetic raw dataset loaded\n")

for k, v in dataset.items():
    print(f"{k:<10} {v.shape}")


# slope function

def calc_slope(x):

    idx = np.arange(len(x))

    if len(x) < 2:
        return 0.0

    p = Polynomial.fit(
        idx,
        x,
        1
    )

    coef = p.convert().coef

    if len(coef) < 2:
        return 0.0

    return coef[1]


# feature extraction

stats_functions = {
    "mean": np.mean,
    "std": np.std,
    "skew": skew,
    "kurtosis": kurtosis,
    "slope": calc_slope,
    "max": np.max
}

features = []


for sensor, df in dataset.items():
    if sensor == "profile":
        continue
    print(f"\nextracting features from {sensor}")
    for stat_name, func in stats_functions.items():
        col_name = f"{sensor}_{stat_name}"
        stat_values = df.apply(func, axis=1)
        features.append(stat_values.rename(col_name))

X_features = pd.concat(
    features,
    axis=1
)


# profile

profile_columns = [
    "cooler_condition",
    "valve_condition",
    "internal_pump_leakage",
    "hydraulic_accumulator",
    "stable_flag"
]

profile_df = dataset["profile"].copy()

profile_df.columns = profile_columns

X_full = pd.concat(
    [X_features, profile_df],
    axis=1
)

print("\nfeature matrix created\n")
print(X_full.shape)


# load scaler

scaler = joblib.load(
    os.path.join(
        models_dir,
        "standard_scaler.pkl"
    )
)

print("\nscaler loaded")


# model config

model_configs = {

    "cooler_condition": {
        "model": "cooler_condition_model.pkl",
        "features": "cooler_condition_features.pkl",
        "label_encoder": None
    },

    "valve_condition": {
        "model": "valve_condition_model.pkl",
        "features": "valve_condition_features.pkl",
        "label_encoder": "valve_condition_label_encoder.pkl"
    },

    "internal_pump_leakage": {
        "model": "internal_pump_leakage_model.pkl",
        "features": "internal_pump_leakage_features.pkl",
        "label_encoder": "internal_pump_leakage_label_encoder.pkl"
    },

    "hydraulic_accumulator": {
        "model": "hydraulic_accumulator_model.pkl",
        "features": "hydraulic_accumulator_features.pkl",
        "label_encoder": "hydraulic_accumulator_label_encoder.pkl"
    },

    "stable_flag": {
        "model": "stable_flag_model.pkl",
        "features": "stable_flag_features.pkl",
        "label_encoder": "stable_flag_label_encoder.pkl"
    }
}


# store metrics

results = []


# evaluation loop

for target, config in model_configs.items():

    print(f"\nevaluating: {target}")

    # load model

    model = joblib.load(
        os.path.join(
            models_dir,
            config["model"]
        )
    )

    # load feature names

    feature_names = joblib.load(
        os.path.join(
            models_dir,
            config["features"]
        )
    )

    # build x and y

    X = X_full[
        feature_names
    ].copy()

    y_true = X_full[
        target
    ].copy()

    # create full scaler-compatible dataframe

    X_complete = pd.DataFrame(
        np.zeros(
            (
                len(X),
                len(scaler.feature_names_in_)
            )
        ),
        columns=scaler.feature_names_in_
    )

    X_complete[
        feature_names
    ] = X

    # scale

    X_scaled_all = scaler.transform(
        X_complete
    )

    X_scaled_all = pd.DataFrame(
        X_scaled_all,
        columns=scaler.feature_names_in_
    )

    X_scaled = X_scaled_all[
        feature_names
    ]

    # predictions

    y_pred = model.predict(
        X_scaled
    )

    # label decoding

    if config["label_encoder"] is not None:

        le = joblib.load(
            os.path.join(
                models_dir,
                config["label_encoder"]
            )
        )

        # true labels already exist
        # in real-world values

        y_true_eval = y_true.astype(int)

        # decode model predictions

        y_pred_eval = le.inverse_transform(
            y_pred.astype(int)
        )

    else:

        y_true_eval = y_true

        y_pred_eval = y_pred

    # metrics

    accuracy = accuracy_score(
        y_true_eval,
        y_pred_eval
    )

    precision = precision_score(
        y_true_eval,
        y_pred_eval,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true_eval,
        y_pred_eval,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true_eval,
        y_pred_eval,
        average="weighted",
        zero_division=0
    )

    # save metrics

    results.append({
        "target": target,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # reports

    print("\nclassification report\n")

    print(
        classification_report(
            y_true_eval,
            y_pred_eval,
            zero_division=0
        )
    )

    print("metrics\n")

    print(f"accuracy : {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"f1-score : {f1:.4f}")


# final results

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)

print("\nfinal results\n")

print(results_df)

results_df.to_csv(
    os.path.join(
        synthetic_dir,
        "synthetic_model_results.csv"
    ),
    index=False
)

print("\nresults saved")