import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from osl_dynamics.models import mdynemo
from osl_dynamics.inference import tf_ops, modes
from osl_dynamics.utils.misc import load, save
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power, connectivity, statistics, regression
from osl_dynamics import run_pipeline

tf_ops.gpu_growth()


def get_best_run(output_dir):
    model_dir_list = os.listdir(output_dir)
    history_file_list = [f"{d}/model/history.pkl" for d in model_dir_list]

    best_loss = np.Inf
    for i, f in enumerate(history_file_list):
        if os.path.exists(f):
            with open(f, "rb") as file:
                history = pickle.load(file)
            if history["loss"][-1] < best_loss:
                best_loss = history["loss"][-1]
                best_run = i
    return best_run


def save_inf_params(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    best_run = get_best_run(output_dir)
    model = mdynemo.Model.load(f"{output_dir}/{best_run:02d}/model")

    alpha, beta = model.get_mode_time_courses(data)
    _, stds, corrs = model.get_means_stds_corrs()

    save(f"{inf_params_dir}/alp.pkl", alpha)
    save(f"{inf_params_dir}/bet.pkl", beta)
    save(f"{inf_params_dir}/stds.npy", stds)
    save(f"{inf_params_dir}/corrs.npy", corrs)


def save_mtc(data, output_dir):
    figures_dir = f"{output_dir}/best_run/figures"
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")

    norm_alpha = modes.reweight_mtc(alpha, np.square(stds), "covariance")
    norm_beta = modes.reweight_mtc(beta, corrs, "correlation")
    plotting.plot_alpha(
        norm_alpha[0],
        norm_beta[0],
        n_samples=2000,
        sampling_frequency=data.sampling_frequency,
        cmap="tab10",
        filename=f"{figures_dir}/renorm_mtc.png",
    )


def save_networks(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")

    n_modes = stds.shape[0]
    n_corr_modes = corrs.shape[0]

    power.save(
        np.square(stds),
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        filename=f"{figures_dir}/vars.png",
        combined=True,
        titles=[f"mode {i+1}" for i in range(n_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    )

    thres_corrs = connectivity.threshold(
        corrs,
        percentile=90,
        absolute_value=True,
    )
    connectivity.save(
        thres_corrs,
        parcellation_file=data.parcellation_file,
        combined=True,
        titles=[f"mode {i+1}" for i in range(n_corr_modes)],
        filename=f"{figures_dir}/corrs.png",
    )


def plot_mode_coupling(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")

    corr = [np.corrcoef(a, b, rowvar=False) for a, b in zip(alpha, beta)]
    g_corr = np.nanmean(corr, axis=0)

    n_modes = alpha[0].shape[1]
    n_corr_modes = beta[0].shape[1]
    m, n = np.tril_indices(n_modes + n_corr_modes, -1)
    flattened_corr = np.array([c[m, n] for c in corr])
    flattened_p = np.squeeze(
        statistics.evoked_response_max_stat_perm(
            flattened_corr[..., None],
            n_perm=1000,
            n_jobs=12,
        )
    )
    p = np.eye(n_modes + n_corr_modes)
    p[m, n] = flattened_p
    p[n, m] = flattened_p

    sig_labels = np.where(p < 0.05, "*", "")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    vmax = 1
    sns.heatmap(
        g_corr[:n_modes, :n_modes],
        annot=sig_labels[:n_modes, :n_modes],
        fmt="s",
        cmap="coolwarm",
        ax=ax[0],
        vmin=-vmax,
        vmax=vmax,
    )
    ax[0].set_title(r"$\alpha - \alpha$", fontsize=22)
    sns.heatmap(
        g_corr[n_corr_modes:, n_corr_modes:],
        annot=sig_labels[n_corr_modes:, n_corr_modes:],
        fmt="s",
        cmap="coolwarm",
        ax=ax[1],
        vmin=-vmax,
        vmax=vmax,
    )
    ax[1].set_title(r"$\beta - \beta$", fontsize=22)
    sns.heatmap(
        g_corr[:n_modes, n_corr_modes:],
        annot=sig_labels[:n_modes, n_corr_modes:],
        fmt="s",
        cmap="coolwarm",
        ax=ax[2],
        vmin=-vmax,
        vmax=vmax,
    )
    ax[2].set_title(r"$\alpha - \beta$", fontsize=22)
    for index, a in enumerate(ax):
        a.set_xticklabels(range(1, 5))
        a.set_yticklabels(range(1, 5))
        if index == 2:
            a.set_xlabel(r"$\beta$ mode", fontsize=16)
            a.set_ylabel(r"$\alpha$ mode", fontsize=16)
        elif index == 1:
            a.set_xlabel(r"$\beta$ mode", fontsize=16)
            a.set_ylabel(r"$\beta$ mode", fontsize=16)
        else:
            a.set_xlabel(r"$\alpha$ mode", fontsize=16)
            a.set_ylabel(r"$\alpha$ mode", fontsize=16)
    fig.suptitle(
        "Correlation profile of M-DyNeMo inferred mode time courses", fontsize=24
    )
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/mode_coupling.png")


def regress_on_alpha(data, output_dir, window_length, step_size):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")

    data.prepare(
        {
            "filter": {"low_freq": 1, "high_freq": 45, "use_raw": True},
            "amplitude_envelope": {},
            "moving_average": {"n_window": 25},
            "standardize": {},
        }
    )
    time_series = data.time_series()
    for i in range(data.n_sessions):
        time_series[i] = time_series[i][: len(alpha[i])]

    sw_cov = connectivity.sliding_window_connectivity(
        time_series,
        window_length=window_length,
        step_size=step_size,
        conn_type="cov",
        n_jobs=12,
    )
    sw_alpha = power.sliding_window_power(
        alpha,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
    )
    regress_covs = []
    for i in range(data.n_sessions):
        regress_covs.append(
            regression.linear(
                sw_alpha[i],
                sw_cov[i],
                fit_intercept=False,
            )
        )
    regress_covs = np.array(regress_covs)
    g_regress_covs = np.nanmean(regress_covs, axis=0)

    n_modes = g_regress_covs.shape[0]
    power.save(
        g_regress_covs,
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        combined=True,
        titles=[f"Mode {i+1}" for i in range(n_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"{figures_dir}/regress_var.png",
    )

    thres_regress_covs = connectivity.threshold(
        g_regress_covs,
        percentile=90,
        absolute_value=True,
    )
    connectivity.save(
        thres_regress_covs,
        parcellation_file=data.parcellation_file,
        combined=True,
        titles=[f"Mode {i+1}" for i in range(n_modes)],
        filename=f"{figures_dir}/regress_cov.png",
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
            store_dir: tmp/notts_38_mdynemo
        prepare:
            filter: {low_freq: 1, high_freq: 45}
            amplitude_envelope: {}
            moving_average: {n_window: 25}
            standardize: {}
            pca: {n_pca_components: 38}
    save_inf_params: {}
    save_mtc: {}
    save_networks: {}
    plot_mode_coupling: {}
    regress_on_alpha:
        window_length: 500
        step_size: 25
"""

run_pipeline(
    config,
    output_dir=f"results/notts_38_mdynemo",
    extra_funcs=[
        save_inf_params,
        save_mtc,
        save_networks,
        plot_mode_coupling,
        regress_on_alpha,
    ],
)
