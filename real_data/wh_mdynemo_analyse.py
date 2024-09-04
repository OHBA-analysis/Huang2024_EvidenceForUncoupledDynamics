"""
Script for analysing and plotting results from training M-DyNeMo
on the Wakemen-Henson dataset.

This script includes the following steps:
1. save the inferred parameters.
2. plot the mode time courses.
3. plot the networks.
4. plot the mode coupling (correlation profile of the mode time courses).
5. Perform evoked network analysis.
6. Permutation test on the spatial map similarity between power and FC networks.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import mne

from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.data import Data
from osl_dynamics.utils.misc import load
from osl_dynamics.analysis import statistics

from helper_functions import (
    save_inf_params,
    plot_mtc,
    plot_networks,
    plot_mode_coupling,
    null_spatial_map_similarity,
    plot_spatial_map_similarity,
)

tf_ops.gpu_growth()


def load_data():
    data = Data(
        sorted(
            glob(
                "/well/woolrich/projects/wakeman_henson/spring23/src/sub*_run*/sflip_parc-raw.fif"
            )
        ),
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
        sampling_frequency=250,
        picks="misc",
        reject_by_annotation="omit",
        n_jobs=12,
        store_dir="tmp/wh_mdynemo",
    )
    data.prepare(
        {
            "filter": {"low_freq": 1, "high_freq": 45, "use_raw": True},
            "amplitude_envelope": {},
            "moving_average": {"n_window": 25},
            "standardize": {},
            "pca": {"n_pca_components": data.n_channels},
        }
    )
    return data


def save_epochs(output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    epochs_dir = f"{output_dir}/best_run/epochs"
    os.makedirs(epochs_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")

    norm_alpha = modes.reweight_mtc(alpha, np.square(stds), "covariance")
    norm_beta = modes.reweight_mtc(beta, corrs, "correlation")

    parc_files = sorted(
        glob(
            "/well/woolrich/projects/wakeman_henson/spring23/src/sub*_run*/sflip_parc-raw.fif"
        )
    )
    new_event_ids = {"famous": 1, "unfamiliar": 2, "scrambled": 3}
    old_event_ids = {
        "famous": [5, 6, 7],
        "unfamiliar": [13, 14, 15],
        "scrambled": [17, 18, 19],
    }
    for a, b, p in zip(norm_alpha, norm_beta, parc_files):
        # Create MNE raw objects
        raw_a = modes.convert_to_mne_raw(a, p, n_window=25)
        raw_b = modes.convert_to_mne_raw(b, p, n_window=25)

        # Find events
        events_a = mne.find_events(raw_a, min_duration=0.005, verbose=False)
        events_b = mne.find_events(raw_b, min_duration=0.005, verbose=False)
        for old_event_codes, new_event_codes in zip(
            old_event_ids.values(), new_event_ids.values()
        ):
            events_a = mne.merge_events(events_a, old_event_codes, new_event_codes)
            events_b = mne.merge_events(events_b, old_event_codes, new_event_codes)

        # Epoch
        epochs_a = mne.Epochs(
            raw_a,
            events_a,
            new_event_ids,
            tmin=-0.1,
            tmax=1.5,
        )
        epochs_b = mne.Epochs(
            raw_b,
            events_b,
            new_event_ids,
            tmin=-0.1,
            tmax=1.5,
        )

        # Save
        id = p.split("/")[-2]
        epochs_a.save(f"{epochs_dir}/{id}-alp_epo.fif", overwrite=True)
        epochs_b.save(f"{epochs_dir}/{id}-bet_epo.fif", overwrite=True)


def first_level_analysis(output_dir):
    epochs_dir = f"{output_dir}/best_run/epochs"
    first_level_dir = f"{output_dir}/best_run/first_level"
    os.makedirs(first_level_dir, exist_ok=True)

    def save_contrasts(name, famous, unfamiliar, scrambled):
        contrasts = [
            (famous + unfamiliar + scrambled) / 3,
            famous + unfamiliar - 2 * scrambled,
            famous - unfamiliar,
        ]
        for i, contrast in enumerate(contrasts):
            filename = f"{first_level_dir}/{name}_contrast_{i}.npy"
            np.save(filename, contrast)

    for a_file, b_file in zip(
        sorted(glob(f"{epochs_dir}/*-alp_epo.fif")),
        sorted(glob(f"{epochs_dir}/*-bet_epo.fif")),
    ):
        id = a_file.split("/")[-1].split("-")[0]
        a_epochs = mne.read_epochs(a_file, verbose=False).pick("misc")
        b_epochs = mne.read_epochs(b_file, verbose=False).pick("misc")

        a_trials, b_trials = {}, {}
        for name in a_epochs.event_id:
            a_trials[name] = a_epochs[name].get_data()
            b_trials[name] = b_epochs[name].get_data()

        a_famous = np.mean(a_trials["famous"], axis=0)
        a_unfamiliar = np.mean(a_trials["unfamiliar"], axis=0)
        a_scrambled = np.mean(a_trials["scrambled"], axis=0)
        save_contrasts(f"alp-{id}", a_famous, a_unfamiliar, a_scrambled)

        b_famous = np.mean(b_trials["famous"], axis=0)
        b_unfamiliar = np.mean(b_trials["unfamiliar"], axis=0)
        b_scrambled = np.mean(b_trials["scrambled"], axis=0)
        save_contrasts(f"bet-{id}", b_famous, b_unfamiliar, b_scrambled)

    np.save(f"{first_level_dir}/t.npy", a_epochs.times)


def group_level_analysis(output_dir):
    first_level_dir = f"{output_dir}/best_run/first_level"
    group_level_dir = f"{output_dir}/best_run/group_level"
    os.makedirs(group_level_dir, exist_ok=True)

    t = np.load(f"{first_level_dir}/t.npy")
    for contrast in range(3):
        a_first_level_files = sorted(
            glob(f"{first_level_dir}/alp-*_contrast_{contrast}.npy")
        )
        b_first_level_files = sorted(
            glob(f"{first_level_dir}/bet-*_contrast_{contrast}.npy")
        )

        a_epochs = np.array([np.load(f) for f in a_first_level_files])
        b_epochs = np.array([np.load(f) for f in b_first_level_files])

        # Baseline correct
        a_epochs -= np.mean(a_epochs[..., t < 0], axis=-1, keepdims=True)
        b_epochs -= np.mean(b_epochs[..., t < 0], axis=-1, keepdims=True)

        a_pvalues = statistics.evoked_response_max_stat_perm(
            a_epochs, n_perm=1000, n_jobs=16
        )
        b_pvalues = statistics.evoked_response_max_stat_perm(
            b_epochs, n_perm=1000, n_jobs=16
        )

        a_epochs = np.mean(a_epochs, axis=0)
        b_epochs = np.mean(b_epochs, axis=0)

        np.save(f"{group_level_dir}/alp-contrast_{contrast}.npy", a_epochs)
        np.save(f"{group_level_dir}/bet-contrast_{contrast}.npy", b_epochs)
        np.save(f"{group_level_dir}/alp-contrast_{contrast}_pvalues.npy", a_pvalues)
        np.save(f"{group_level_dir}/bet-contrast_{contrast}_pvalues.npy", b_pvalues)


def evoked_response_analysis(data, output_dir):
    cmap = plt.get_cmap("tab10")
    contrasts = ["visual", "faces_vs_scrambled", "famous_vs_unfamiliar"]

    save_epochs(output_dir)
    first_level_analysis(output_dir)
    group_level_analysis(output_dir)

    first_level_dir = f"{output_dir}/best_run/first_level"
    group_level_dir = f"{output_dir}/best_run/group_level"
    figures_dir = f"{output_dir}/best_run/figures"

    t = np.load(f"{first_level_dir}/t.npy")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for index, name in enumerate(contrasts):
        # Load COPEs from the group-level GLM
        cope = np.load(f"{group_level_dir}/alp-contrast_{index}.npy")
        pvalues = np.load(f"{group_level_dir}/alp-contrast_{index}_pvalues.npy")

        # Plot
        for mode in range(cope.shape[0]):
            # Plot group mean
            axes[0, index].plot(
                t,
                cope[mode],
                label=f"Mode {mode+1}" if index == 0 else None,
                linewidth=3,
                color=cmap(mode),
            )

            # Add a bar showing significant time point
            sig_t = t[pvalues[mode] < 0.05]
            if len(sig_t) > 0:
                dt = t[1] - t[0]
                y = cope.max() * (1.3 + 0.1 * mode)
                for st in sig_t:
                    axes[0, index].plot(
                        (st - dt, st + dt), (y, y), color=cmap(mode), linewidth=4
                    )

        # Tidy up plot
        # axes[0, index].set_xlabel("Time (s)", fontsize=16)
        axes[0, index].set_title(name, fontsize=22)
        axes[0, index].tick_params(axis="both", labelsize=16)
        axes[0, index].set_xlim(t[0], t[-1])
    axes[0, 0].set_ylabel(r"$\alpha$ Activation", fontsize=22)

    for index, name in enumerate(contrasts):
        # Load COPEs from the group-level GLM
        cope = np.load(f"{group_level_dir}/bet-contrast_{index}.npy")
        pvalues = np.load(f"{group_level_dir}/bet-contrast_{index}_pvalues.npy")

        for mode in range(cope.shape[0]):
            # Plot group mean
            axes[1, index].plot(t, cope[mode], linewidth=3, color=cmap(mode))

            # Add a bar showing significant time point
            sig_t = t[pvalues[mode] < 0.05]
            if len(sig_t) > 0:
                dt = t[1] - t[0]
                y = cope.max() * (1.3 + 0.1 * mode)
                for st in sig_t:
                    axes[1, index].plot(
                        (st - dt, st + dt), (y, y), color=cmap(mode), linewidth=4
                    )

        # Tidy up plot
        axes[1, index].set_xlabel("Time (s)", fontsize=22)
        axes[1, index].tick_params(axis="both", labelsize=16)
        axes[1, index].set_xlim(t[0], t[-1])
    axes[1, 0].set_ylabel(r"$\beta$ Activation", fontsize=22)
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
        fontsize=20,
    )
    fig.suptitle("COPEs for evoked responses", fontsize=28)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/evoked_response.png")


data = load_data()

config = """
    save_inf_params: {}
    plot_mtc: {}
    plot_mode_coupling: {}
    plot_networks: {}
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
    output_dir="results/wh_mdynemo",
    data=data,
    extra_funcs=[
        save_inf_params,
        plot_mtc,
        plot_mode_coupling,
        plot_networks,
        evoked_response_analysis,
        null_spatial_map_similarity,
        plot_spatial_map_similarity,
    ],
)
