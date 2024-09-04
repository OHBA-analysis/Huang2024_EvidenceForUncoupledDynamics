"""
Script for analysing and plotting results from running DyNeMo
on MEGUK-38 resting-state data.

This script includes the following steps:
1. save the inferred parameters.
2. plot the mode time courses.
3. plot the networks.
"""

import os

import numpy as np

from osl_dynamics.models import dynemo
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.utils.misc import load, save
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power, connectivity
from osl_dynamics import run_pipeline

from helper_functions import get_best_run

tf_ops.gpu_growth()


def save_inf_params(data, output_dir):
    """Save the inferred parameters."""
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    best_run = get_best_run(output_dir)
    model = dynemo.Model.load(f"{output_dir}/{best_run}/model")

    alpha = model.get_alpha(data)
    covs = model.get_covariances()

    save(f"{inf_params_dir}/alp.pkl", alpha)
    save(f"{inf_params_dir}/covs.npy", covs)


def plot_mtc(data, output_dir, order=None):
    """Plot the mode time courses.

    Parameters
    ----------
    order : array-like, optional
        The order of the modes. If None, the order is not changed.
    """
    figures_dir = f"{output_dir}/best_run/figures"
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    covs = load(f"{inf_params_dir}/covs.npy")

    # Order the modes
    if order is None:
        n_modes = covs.shape[0]
        order = np.arange(n_modes)

    alpha = [a[:, order] for a in alpha]
    covs = covs[order]

    norm_alpha = modes.reweight_mtc(alpha, covs, "covariance")
    plotting.plot_alpha(
        norm_alpha[0],
        n_samples=2000,
        sampling_frequency=data.sampling_frequency,
        cmap="tab10",
        filename=f"{figures_dir}/renorm_mtc.png",
    )


def plot_networks(data, output_dir, order=None):
    """Plot the networks.

    Parameters
    ----------
    order : array-like, optional
        The order of the modes. If None, the order is not changed.
    """
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    covs = load(f"{inf_params_dir}/covs.npy")
    real_covs = data.pca_components @ covs @ data.pca_components.T
    n_modes = covs.shape[0]

    # Order the modes
    if order is None:
        order = np.arange(n_modes)
    real_covs = real_covs[order]

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


config = """
    load_data:
        inputs: /well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks
        kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 8
            use_tfrecord: True
            buffer_size: 2000
            store_dir: tmp/notts_38_dynemo
        prepare:
            filter: {low_freq: 1, high_freq: 45}
            amplitude_envelope: {}
            moving_average: {n_window: 25}
            standardize: {}
            pca: {n_pca_components: 38}
    save_inf_params: {}
    plot_mtc: {}
    plot_networks: {}
"""

run_pipeline(
    config,
    output_dir=f"results/notts_38_dynemo",
    extra_funcs=[save_inf_params, plot_mtc, plot_networks],
)
