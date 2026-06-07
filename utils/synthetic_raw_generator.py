import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis


# load raw dataset

def load_raw_dataset(data_dir, files):
    """
    load all raw sensor txt files.
    """

    dataset = {}

    for file in files:

        path = os.path.join(data_dir, file)

        df = pd.read_csv(
            path,
            sep="\t",
            header=None
        )

        sensor_name = file.replace(".txt", "")

        dataset[sensor_name] = df

    return dataset


# feature preservation helpers

def recenter_mean(signal, original):
    """
    force synthetic signal to keep original mean.
    """

    return (
        signal
        - np.mean(signal)
        + np.mean(original)
    )


def correct_std(signal, original):
    """
    force synthetic signal to keep original std.
    """

    orig_std = np.std(original)
    synt_std = np.std(signal)

    if synt_std < 1e-10:
        return signal.copy()

    signal = (
        np.mean(original)
        + (
            signal - np.mean(signal)
        ) * (
            orig_std / synt_std
        )
    )

    return signal


# temporal warp

def temporal_warp(
    signal,
    warp_strength=0.0015
):
    """
    apply tiny temporal deformation.

    this is the safest augmentation because
    it preserves the amplitude distribution.
    """

    n = len(signal)

    x_original = np.linspace(
        0,
        1,
        n
    )

    noise = gaussian_filter1d(
        np.random.normal(
            0,
            warp_strength,
            n
        ),
        sigma=25
    )

    x_warped = x_original + noise

    x_warped = np.clip(
        x_warped,
        0,
        1
    )

    x_warped = np.sort(
        x_warped
    )

    warped = np.interp(
        x_original,
        x_warped,
        signal
    )

    return warped


# slow drift

def add_slow_drift(
    signal,
    drift_strength=0.0005
):
    """
    tiny baseline drift.
    """

    drift = np.linspace(
        np.random.uniform(
            -drift_strength,
            drift_strength
        ),
        np.random.uniform(
            -drift_strength,
            drift_strength
        ),
        len(signal)
    )

    return signal + drift


# interpolation augmentation

def interpolate_signals(
    signal_a,
    signal_b,
    alpha=None
):
    """
    interpolate between two real signals.

    this preserves physical realism much better
    than gaussian noise.
    """

    if alpha is None:

        alpha = np.random.uniform(
            0.92,
            0.98
        )

    synthetic = (
        alpha * signal_a
        + (1 - alpha) * signal_b
    )

    return synthetic


# validation

def validate_signal(
    original,
    synthetic,
    sensor_name=""
):

    stats = {
        "mean": (
            np.mean(original),
            np.mean(synthetic)
        ),
        "std": (
            np.std(original),
            np.std(synthetic)
        ),
        "skew": (
            skew(original),
            skew(synthetic)
        ),
        "kurtosis": (
            kurtosis(original),
            kurtosis(synthetic)
        )
    }

    print(f"\nvalidation -> {sensor_name}")

    for name, (orig, synth) in stats.items():

        delta = (
            abs(synth - orig)
            / (abs(orig) + 1e-8)
        ) * 100

        print(
            f"{name:<10} "
            f"orig={orig:>10.5f} "
            f"synth={synth:>10.5f} "
            f"delta={delta:>8.3f}%"
        )


# generate synthetic dataset

def generate_synthetic_dataset(
    dataset,
    n_samples=20,
    random_state=42,
    validate=False
):
    """
    generate synthetic cycles preserving:
    - temporal structure
    - cross-sensor consistency
    - statistical moments
    """

    np.random.seed(random_state)

    synthetic_dataset = {}

    profile_df = dataset["profile"]

    sensors = [
        s for s in dataset.keys()
        if s != "profile"
    ]

    synthetic_signals = {
        sensor: []
        for sensor in sensors
    }

    synthetic_profiles = []

    # group indices by full profile row
    grouped_indices = {}

    for idx, row in profile_df.iterrows():

        key = tuple(row.values)

        if key not in grouped_indices:
            grouped_indices[key] = []

        grouped_indices[key].append(idx)

    class_keys = list(
        grouped_indices.keys()
    )

    # generate cycles
    for sample_idx in range(n_samples):

        # pick one operating condition
        class_key = class_keys[
            np.random.randint(
                0,
                len(class_keys)
            )
        ]

        candidate_indices = grouped_indices[
            class_key
        ]

        # choose two real cycles
        idx_a = np.random.choice(
            candidate_indices
        )

        idx_b = np.random.choice(
            candidate_indices
        )

        synthetic_profiles.append(
            class_key
        )

        # shared augmentation params
        shared_warp = np.random.uniform(
            0.0005,
            0.002
        )

        shared_drift = np.random.uniform(
            0.0001,
            0.0008
        )

        shared_alpha = np.random.uniform(
            0.93,
            0.985
        )

        # apply same params to all sensors
        for sensor in sensors:

            signal_a = dataset[sensor].iloc[
                idx_a
            ].values.astype(float)

            signal_b = dataset[sensor].iloc[
                idx_b
            ].values.astype(float)

            synthetic = interpolate_signals(
                signal_a,
                signal_b,
                alpha=shared_alpha
            )

            synthetic = temporal_warp(
                synthetic,
                warp_strength=shared_warp
            )

            synthetic = add_slow_drift(
                synthetic,
                drift_strength=shared_drift
            )

            synthetic = recenter_mean(
                synthetic,
                signal_a
            )

            synthetic = correct_std(
                synthetic,
                signal_a
            )

            synthetic_signals[sensor].append(
                synthetic
            )

            if (
                validate
                and sample_idx == 0
            ):
                validate_signal(
                    signal_a,
                    synthetic,
                    sensor
                )

    # convert to dfs
    for sensor in sensors:

        synthetic_dataset[sensor] = pd.DataFrame(
            synthetic_signals[sensor]
        )

    synthetic_dataset["profile"] = pd.DataFrame(
        synthetic_profiles,
        columns=profile_df.columns
    )

    return synthetic_dataset


# save synthetic dataset

def save_synthetic_dataset(
    synthetic_dataset,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for sensor, df in synthetic_dataset.items():

        path = os.path.join(
            output_dir,
            f"{sensor}.txt"
        )

        df.to_csv(
            path,
            sep="\t",
            header=False,
            index=False
        )

        print(f"saved -> {path}")


# complete pipeline

def create_synthetic_raw_dataset(
    data_dir,
    output_dir,
    files,
    n_samples=300,
    random_state=42,
    validate=False
):

    print("\nloading raw dataset")

    dataset = load_raw_dataset(
        data_dir=data_dir,
        files=files
    )

    print("\ngenerating synthetic dataset")

    synthetic_dataset = generate_synthetic_dataset(
        dataset=dataset,
        n_samples=n_samples,
        random_state=random_state,
        validate=validate
    )

    print("\nsaving synthetic dataset")

    save_synthetic_dataset(
        synthetic_dataset=synthetic_dataset,
        output_dir=output_dir
    )

    print("\nsynthetic dataset generated successfully")


# execution

BASE_DIR = Path.cwd()

data_dir = (
    BASE_DIR
    / "database"
    / "bronce"
)

output_dir = (
    BASE_DIR
    / "database"
    / "synthetic"
)

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

create_synthetic_raw_dataset(
    data_dir=data_dir,
    output_dir=output_dir,
    files=files,
    n_samples=20,
    random_state=42,
    validate=False
)