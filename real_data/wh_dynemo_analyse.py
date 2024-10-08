"""
Script for analysing and plotting results from running DyNeMo
on the Wakeman-Henson dataset.

The script includes the following steps:
1. save the inferred parameters.
2. plot the mode time courses.
3. plot the networks.
4. Perform evoked network analysis.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import mne

from osl_dynamics import run_pipeline
from osl_dynamics.models import dynemo
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.data import Data
from osl_dynamics.utils.misc import load, save
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power, connectivity, statistics

from helper_functions import get_best_run

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
        store_dir="tmp/wh_dynemo",
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


def save_inf_params(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    best_run = get_best_run(output_dir)
    model = dynemo.Model.load(f"{output_dir}/{best_run}/model")

    alpha = model.get_alpha(data)
    covs = model.get_covariances()

    save(f"{inf_params_dir}/alp.pkl", alpha)
    save(f"{inf_params_dir}/covs.npy", covs)


def plot_mtc(data, output_dir):
    figures_dir = f"{output_dir}/best_run/figures"
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    covs = load(f"{inf_params_dir}/covs.npy")

    norm_alpha = modes.reweight_mtc(alpha, covs, "covariance")
    plotting.plot_alpha(
        norm_alpha[0],
        n_samples=2000,
        sampling_frequency=data.sampling_frequency,
        cmap="tab10",
        filename=f"{figures_dir}/renorm_mtc.png",
    )


def plot_networks(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    covs = load(f"{inf_params_dir}/covs.npy")
    real_covs = data.pca_components @ covs @ data.pca_components.T

    n_modes = covs.shape[0]
    power.save(
        real_covs,
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        filename=f"{figures_dir}/var.png",
        combined=True,
        titles=[f"DyNeMo mode {i+1}" for i in range(n_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    )
    thres_real_covs = connectivity.threshold(
        real_covs - np.mean(real_covs, axis=0),
        percentile=90,
        absolute_value=True,
        subtract_mean=True,
    )
    connectivity.save(
        thres_real_covs,
        parcellation_file=data.parcellation_file,
        combined=True,
        filename=f"{figures_dir}/cov.png",
    )


def save_epochs(output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    epochs_dir = f"{output_dir}/best_run/epochs"
    os.makedirs(epochs_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    covs = load(f"{inf_params_dir}/covs.npy")

    norm_alpha = modes.reweight_mtc(alpha, covs, "covariance")

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
    for a, p in zip(norm_alpha, parc_files):
        # Create MNE raw objects
        raw_a = modes.convert_to_mne_raw(a, p, n_window=25)

        # Find events
        events_a = mne.find_events(raw_a, min_duration=0.005, verbose=False)
        for old_event_codes, new_event_codes in zip(
            old_event_ids.values(), new_event_ids.values()
        ):
            events_a = mne.merge_events(events_a, old_event_codes, new_event_codes)

        # Epoch
        epochs_a = mne.Epochs(
            raw_a,
            events_a,
            new_event_ids,
            tmin=-0.1,
            tmax=1.5,
        )

        # Save
        id = p.split("/")[-2]
        epochs_a.save(f"{epochs_dir}/{id}-alp_epo.fif", overwrite=True)


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

    for a_file in sorted(glob(f"{epochs_dir}/*-alp_epo.fif")):
        id = a_file.split("/")[-1].split("-")[0]
        a_epochs = mne.read_epochs(a_file, verbose=False).pick("misc")

        a_trials = {}
        for name in a_epochs.event_id:
            a_trials[name] = a_epochs[name].get_data()

        a_famous = np.mean(a_trials["famous"], axis=0)
        a_unfamiliar = np.mean(a_trials["unfamiliar"], axis=0)
        a_scrambled = np.mean(a_trials["scrambled"], axis=0)
        save_contrasts(f"alp-{id}", a_famous, a_unfamiliar, a_scrambled)

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
        a_epochs = np.array([np.load(f) for f in a_first_level_files])

        # Baseline correct
        a_epochs -= np.mean(a_epochs[..., t < 0], axis=-1, keepdims=True)

        a_pvalues = statistics.evoked_response_max_stat_perm(
            a_epochs, n_perm=1000, n_jobs=16
        )
        a_epochs = np.mean(a_epochs, axis=0)

        np.save(f"{group_level_dir}/alp-contrast_{contrast}.npy", a_epochs)
        np.save(f"{group_level_dir}/alp-contrast_{contrast}_pvalues.npy", a_pvalues)


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

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for index, name in enumerate(contrasts):
        # Load COPEs from the group-level GLM
        cope = np.load(f"{group_level_dir}/alp-contrast_{index}.npy")
        pvalues = np.load(f"{group_level_dir}/alp-contrast_{index}_pvalues.npy")

        # Plot
        for mode in range(cope.shape[0]):
            # Plot group mean
            axes[index].plot(
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
                    axes[index].plot(
                        (st - dt, st + dt), (y, y), color=cmap(mode), linewidth=4
                    )

        # Tidy up plot
        # axes[0, index].set_xlabel("Time (s)", fontsize=16)
        axes[index].set_title(name, fontsize=22)
        axes[index].tick_params(axis="both", labelsize=16)
        axes[index].set_xlim(t[0], t[-1])
    axes[0].set_ylabel(r"$\alpha$ Activation", fontsize=22)
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
    plot_networks: {}
    evoked_response_analysis: {}
"""
run_pipeline(
    config,
    output_dir="results/wh_dynemo",
    data=data,
    extra_funcs=[
        save_inf_params,
        plot_mtc,
        plot_networks,
        evoked_response_analysis,
    ],
)
