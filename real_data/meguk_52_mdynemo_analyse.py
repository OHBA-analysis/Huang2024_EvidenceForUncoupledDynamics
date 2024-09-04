"""
Script for analysing and plotting results from running M-DyNeMo
on MEGUK-52 data.

This script includes the following steps:
1. save the inferred parameters.
2. plot the mode time courses.
3. plot the networks.
4. Compare the correlation profiles of the mode time courses between rest and task.
5. Perform evoked network analysis.
6. Permutation test on the spatial map similarity between power and FC networks.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.utils.misc import load
from osl_dynamics.analysis import statistics
from osl_dynamics import run_pipeline

from helper_functions import (
    save_inf_params,
    plot_mtc,
    plot_networks,
    null_spatial_map_similarity,
    plot_spatial_map_similarity,
)

tf_ops.gpu_growth()


def get_data_df():
    # Helper functions
    def _get_subject(x):
        return os.path.basename(x).split("_")[0].split("-")[1]

    def _get_site(x):
        return _get_subject(x)[:3]

    def _get_task(x):
        return os.path.basename(x).split("_")[1].split("-")[1].split(".")[0]

    def _get_run(x):
        x_split = os.path.basename(x).split("_")
        if len(x_split) == 2:
            return 1
        elif len(x_split) == 3:
            return int(x_split[2].split("-")[1].split(".")[0])

    df = pd.DataFrame()
    data_dir = "/well/woolrich/projects/mrc_meguk/all_sites/npy/src"
    df["data_path"] = sorted(glob(f"{data_dir}/*"))
    df["subject"] = df["data_path"].apply(_get_subject)
    df["site"] = df["data_path"].apply(_get_site)
    df["task"] = df["data_path"].apply(_get_task)
    df["run"] = df["data_path"].apply(_get_run)
    df = df[df["site"] == "not"]
    df = df[df["task"].isin(["resteyesopen", "visuomotor"])]
    return df


def load_data():
    df = get_data_df()
    data = Data(
        df["data_path"].tolist(),
        sampling_frequency=250,
        use_tfrecord=True,
        buffer_size=2000,
        n_jobs=12,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        store_dir=f"tmp/mdynemo",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    )
    data.prepare(
        {
            "amplitude_envelope": {},
            "moving_average": {"n_window": 25},
            "standardize": {},
            "pca": {"n_pca_components": data.n_channels},
        }
    )
    return data


def mode_coupling(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")

    df = get_data_df()

    # Get the rest correlation
    rest_mask = df["task"] == "resteyesopen"
    rest_alpha = [a for a, mask in zip(alpha, rest_mask) if mask]
    rest_beta = [b for b, mask in zip(beta, rest_mask) if mask]
    rest_corr = [np.corrcoef(a, b, rowvar=False) for a, b in zip(rest_alpha, rest_beta)]

    # Get the task correlation
    task_mask = df["task"] == "visuomotor"
    task_alpha = [a for a, mask in zip(alpha, task_mask) if mask]
    task_beta = [b for b, mask in zip(beta, task_mask) if mask]

    event_ids = {"visual": 1}
    a_epochs, b_epochs = [], []
    for i, ind in enumerate(np.where(task_mask)[0]):
        subject = df["subject"].iloc[ind]
        events = np.load(
            f"/well/woolrich/projects/mrc_meguk/all_sites/events/Nottingham/sub-{subject}_task-visuomotor.npy"
        )
        raw = modes.convert_to_mne_raw(
            task_alpha[i],
            f"/well/woolrich/projects/mrc_meguk/all_sites/src/Nottingham/sub-{subject}_task-visuomotor/parc/parc-raw.fif",
            n_window=25,
            verbose=False,
        )
        epochs = mne.Epochs(
            raw, events, event_ids, tmin=-0.5, tmax=2, on_missing="warn", baseline=None
        )
        a_epochs.append(np.concatenate(epochs["visual"].get_data(), axis=1))
        raw = modes.convert_to_mne_raw(
            task_beta[i],
            f"/well/woolrich/projects/mrc_meguk/all_sites/src/Nottingham/sub-{subject}_task-visuomotor/parc/parc-raw.fif",
            n_window=25,
            verbose=False,
        )
        epochs = mne.Epochs(
            raw, events, event_ids, tmin=-0.5, tmax=2, on_missing="warn", baseline=None
        )
        b_epochs.append(np.concatenate(epochs["visual"].get_data(), axis=1))

    task_corr = [
        np.corrcoef(stc1.T, stc2.T, rowvar=False)
        for stc1, stc2 in zip(a_epochs, b_epochs)
    ]

    # Get the corr difference of common subjects
    rest_df = df[rest_mask]
    task_df = df[task_mask]
    rest_subjects = rest_df["subject"].unique()
    task_subjects = task_df["subject"].unique()
    common_subjects = np.intersect1d(rest_subjects, task_subjects)

    cross_corr_diff = []
    for subject in common_subjects:
        rest_cross_corr = rest_corr[np.where(rest_df["subject"] == subject)[0][0]][
            :4, 4:
        ]
        task_cross_corr = task_corr[np.where(task_df["subject"] == subject)[0][0]][
            :4, 4:
        ]
        if np.any(np.isnan(rest_cross_corr)) or np.any(np.isnan(task_cross_corr)):
            continue
        cross_corr_diff.append(task_cross_corr - rest_cross_corr)
    cross_corr_diff = np.array(cross_corr_diff)

    # Paired permutation test
    paired_diff, pvalues = statistics.paired_diff_max_stat_perm(
        cross_corr_diff, n_perm=1000, metric="copes"
    )

    # plot the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(
        np.nanmean(rest_corr, axis=0)[:4, 4:],
        ax=axes[0],
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    axes[0].set_title("Rest", fontsize=16)
    sns.heatmap(
        np.nanmean(task_corr, axis=0)[:4, 4:],
        ax=axes[1],
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    axes[1].set_title("Task", fontsize=16)

    sig_label = np.where(pvalues <= 0.05, "*", "")
    sig_label = np.where(pvalues <= 0.01, "**", sig_label)
    sig_label = np.where(pvalues <= 0.001, "***", sig_label)
    sns.heatmap(
        paired_diff,
        annot=sig_label,
        fmt="s",
        cmap="coolwarm",
        ax=axes[2],
    )
    axes[2].set_title("Task - Rest", fontsize=16)
    for a in axes:
        a.set_xticklabels(range(1, 5))
        a.set_yticklabels(range(1, 5))
        a.set_xlabel(r"$\beta$ mode", fontsize=16)
        a.set_ylabel(r"$\alpha$ mode", fontsize=16)

    fig.suptitle(r"$\alpha$ - $\beta$ correlation profiles", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/mode_coupling.png")


def evoked_response_analysis(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")

    norm_alpha = modes.reweight_mtc(alpha, np.square(stds), "covariance")
    norm_beta = modes.reweight_mtc(beta, corrs, "correlation")

    df = get_data_df()
    event_ids = {"visual": 1}
    task_mask = df["task"] == "visuomotor"

    task_alpha = [a for a, mask in zip(norm_alpha, task_mask) if mask]
    task_beta = [b for b, mask in zip(norm_beta, task_mask) if mask]
    alpha_epochs, beta_epochs = [], []
    for i, ind in enumerate(np.where(task_mask)[0]):
        subject = df["subject"].iloc[ind]
        events = np.load(
            f"/well/woolrich/projects/mrc_meguk/all_sites/events/Nottingham/sub-{subject}_task-visuomotor.npy"
        )
        raw = modes.convert_to_mne_raw(
            task_alpha[i],
            f"/well/woolrich/projects/mrc_meguk/all_sites/src/Nottingham/sub-{subject}_task-visuomotor/parc/parc-raw.fif",
            n_window=25,
            verbose=False,
        )
        epochs = mne.Epochs(
            raw, events, event_ids, tmin=-0.5, tmax=2, on_missing="warn"
        )
        alpha_epochs.append(np.mean(epochs["visual"].get_data(), axis=0))
        raw = modes.convert_to_mne_raw(
            task_beta[i],
            f"/well/woolrich/projects/mrc_meguk/all_sites/src/Nottingham/sub-{subject}_task-visuomotor/parc/parc-raw.fif",
            n_window=25,
            verbose=False,
        )
        epochs = mne.Epochs(
            raw, events, event_ids, tmin=-0.5, tmax=2, on_missing="warn"
        )
        beta_epochs.append(np.mean(epochs["visual"].get_data(), axis=0))

    alpha_epochs = np.array(alpha_epochs)
    alpha_epochs -= np.mean(alpha_epochs[..., epochs.times < 0], axis=-1, keepdims=True)
    beta_epochs = np.array(beta_epochs)
    beta_epochs -= np.mean(beta_epochs[..., epochs.times < 0], axis=-1, keepdims=True)

    alpha_pvalues = statistics.evoked_response_max_stat_perm(
        alpha_epochs,
        n_perm=1000,
        n_jobs=16,
    )
    alpha_epochs_mean = np.mean(alpha_epochs, axis=0)
    beta_pvalues = statistics.evoked_response_max_stat_perm(
        beta_epochs,
        n_perm=1000,
        n_jobs=16,
    )
    beta_epochs_mean = np.mean(beta_epochs, axis=0)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    t = epochs.times
    for mode in range(4):
        axes[0].plot(
            t,
            alpha_epochs_mean[mode],
            label=f"Mode {mode+1}",
            linewidth=3,
            color=cmap(mode),
        )
        sig_t = t[alpha_pvalues[mode] <= 0.05]
        if len(sig_t) > 0:
            dt = t[1] - t[0]
            y = alpha_epochs_mean.max() * (1.3 + 0.1 * mode)
            for st in sig_t:
                axes[0].plot((st - dt, st + dt), (y, y), color=cmap(mode), linewidth=4)

    axes[0].tick_params(axis="both", labelsize=16)
    axes[0].set_xlim(t[0], t[-1])
    axes[0].set_ylabel(r"$\alpha$ Activation", fontsize=22)

    for mode in range(4):
        axes[1].plot(t, beta_epochs_mean[mode], linewidth=3, color=cmap(mode))
        sig_t = t[beta_pvalues[mode] <= 0.05]
        if len(sig_t) > 0:
            dt = t[1] - t[0]
            y = beta_epochs_mean.max() * (1.3 + 0.1 * mode)
            for st in sig_t:
                axes[1].plot((st - dt, st + dt), (y, y), color=cmap(mode), linewidth=4)

    axes[1].tick_params(axis="both", labelsize=16)
    axes[1].set_xlim(t[0], t[-1])
    axes[1].set_ylabel(r"$\beta$ Activation", fontsize=22)

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fancybox=True,
        shadow=True,
        ncol=4,
        fontsize=20,
    )
    fig.suptitle("COPEs for visual task evoked responses", fontsize=28)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/evoked_response.png")


data = load_data()
config = """
    save_inf_params: {}
    plot_mtc: {}
    plot_networks: {}
    mode_coupling: {}
    evoked_response_analysis: {}
    null_spatial_map_similarity:
        window_length: 50
        step_size: 5
        shuffle_window_length: 250
        n_perm: 1000
        n_jobs: 10
    plot_spatial_map_similarity: {}
"""

run_pipeline(
    config,
    output_dir="results/notts_52_mdynemo",
    data=data,
    extra_funcs=[
        save_inf_params,
        plot_mtc,
        plot_networks,
        mode_coupling,
        evoked_response_analysis,
        null_spatial_map_similarity,
        plot_spatial_map_similarity,
    ],
)
