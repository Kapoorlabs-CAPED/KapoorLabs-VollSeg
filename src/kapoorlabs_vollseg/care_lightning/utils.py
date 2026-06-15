import os
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import OmegaConf

from bokeh.palettes import Category10
import itertools


logger = logging.getLogger(__name__)


def get_most_recent_file(file_path, file_pattern):
    ckpt_files = [file for file in os.listdir(file_path) if file.endswith(file_pattern)]

    if len(ckpt_files) > 0:
        # Sort by modification time (most recent first)
        ckpt_files_with_mtime = [
            (file, os.path.getmtime(os.path.join(file_path, file)))
            for file in ckpt_files
        ]
        sorted_ckpt_files = sorted(
            ckpt_files_with_mtime, key=lambda x: x[1], reverse=True
        )
        most_recent_ckpt = sorted_ckpt_files[0][0]
        return os.path.join(file_path, most_recent_ckpt)
    else:
        return None


def load_checkpoint_model(log_path: str):

    ckpt_path = get_most_recent_file(log_path, ".ckpt")

    return ckpt_path


def plot_npz_files_interactive(
    filepaths,
    unwanted_substrings=["gpu", "memory"],
    page_output_dir="metrics",
    save_plots=False,
    show_plots=True,
):
    all_data = {}
    Path(page_output_dir).mkdir(parents=True, exist_ok=True)

    # Load and merge data
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue

        keys = sorted(data.files, key=lambda x: ("epoch" in x, x), reverse=True)
        for key in keys:
            if any(sub in key for sub in unwanted_substrings):
                continue
            data_values = data[key].tolist()
            if key not in all_data:
                all_data[key] = data_values
            else:
                all_data[key]["steps"].extend(data_values["steps"])
                all_data[key]["values"].extend(data_values["values"])

    # Prepare figure layout
    colors = itertools.cycle(Category10[10])
    grouped_keys = list(all_data.keys())
    n_cols = 4
    n_plots = len(grouped_keys)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, key in enumerate(grouped_keys):
        values = all_data[key]
        df = pd.DataFrame.from_dict(values).sort_values("steps")
        color = next(colors)

        ax = axes[i]
        ax.plot(df["steps"].to_numpy(), df["values"].to_numpy(), label=key, color=color)
        ax.scatter(
            df["steps"].to_numpy(), df["values"].to_numpy(), s=4, color=color, alpha=0.3
        )
        ax.set_title(f"{key}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_plots:
        output_path = Path(page_output_dir) / "metrics_all_in_one.png"
        fig.savefig(output_path, dpi=300)
        print(f"Saved all-in-one plot to: {output_path}")
        if show_plots:
            plt.show()

    if show_plots:
        plt.show()


def plot_npz_files(filepaths):
    all_data = {}
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except pickle.UnpicklingError:
            # print(f"Error loading data from {filepath}. Skipping this file.")
            continue

        keys = data.files
        keys = sorted(keys, key=lambda x: ("epoch" in x, x), reverse=True)
        unwanted_substrings = ["step", "gpu", "memory"]
        for idx, key in enumerate(keys):
            if not any(substring in key for substring in unwanted_substrings):
                data_values = data[key].tolist()
                if key not in all_data:
                    all_data[key] = data_values
                else:
                    all_data[key]["steps"].extend(data_values["steps"])
                    all_data[key]["values"].extend(data_values["values"])
    for k, v in all_data.items():
        data_frame = pd.DataFrame.from_dict(all_data[k])
        sns.lineplot(x="steps", y="values", data=data_frame, label=k)
        plt.show()


def normalize_mi_ma(x, mi, ma, eps=1e-20, dtype=np.float32):
    x = x.astype(dtype)
    mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
    ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
    eps = dtype(eps) if np.isscalar(eps) else eps.astype(dtype, copy=False)

    x = (x - mi) / (ma - mi + eps)

    return x


def percentile_norm(x, pmin=1, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalize_in_chunks(image, chunk_steps=50, pmin=1, pmax=99.8, dtype=np.float32):
    """
    Normalize a TZYX image in chunks along the T (time) dimension.

    Args:
        image (np.ndarray): The original TZYX image.
        chunk_steps (int): The number of timesteps to process at a time.
        pmin (float): The lower percentile for normalization.
        pmax (float): The upper percentile for normalization.
        dtype (np.dtype): The data type to cast the normalized image.

    Returns:
        np.ndarray: The normalized image with the same shape as the input.
    """
    # Get the shape of the original image (T, Z, Y, X)
    T = image.shape[0]

    # Create an empty array to hold the normalized image
    normalized_image = np.empty_like(image, dtype=dtype)

    # Process the image in chunks of `chunk_steps` along the T (time) axis
    for t in range(0, T, chunk_steps):
        # Determine the chunk slice, ensuring we don't go out of bounds
        t_end = min(t + chunk_steps, T)

        # Extract the chunk of timesteps to normalize
        chunk = image[t:t_end]

        # Normalize this chunk
        chunk_normalized = percentile_norm(chunk, pmin=pmin, pmax=pmax, dtype=dtype)

        # Replace the corresponding portion with the normalized chunk
        normalized_image[t:t_end] = chunk_normalized

    return normalized_image


def save_config_as_json(config, log_path):
    """Save resolved OmegaConf config as JSON to log_path"""
    config_dict = OmegaConf.to_container(config, resolve=True)

    config_file = Path(log_path) / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Config saved to: {config_file}")
    return config_dict


__all__ = [
    "get_most_recent_file",
    "load_checkpoint_model",
    "plot_npz_files_interactive",
    "plot_npz_files",
    "save_config_as_json",
    "percentile_norm",
    "normalize_mi_ma",
]
