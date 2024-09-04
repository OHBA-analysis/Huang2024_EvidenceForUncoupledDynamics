"""This module contains functions to analyse the results of M-DyNeMo."""

import os
import pickle
from glob import glob

import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from pqdm.threads import pqdm
from tqdm.auto import tqdm

from osl_dynamics.data import Data
from osl_dynamics.inference import modes
from osl_dynamics.utils.misc import load, save
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power, connectivity, regression, statistics
from osl_dynamics.models import mdynemo
from osl_dynamics import array_ops


def get_best_run(output_dir):
    """Get the best run based on the lowest loss.

    Returns
    -------
    best_run : str
        The best run.
    """
    history_file_list = sorted(glob(f"{output_dir}/run*/model/history.pkl"))

    best_loss = np.Inf
    for i, f in enumerate(history_file_list):
        with open(f, "rb") as file:
            history = pickle.load(file)
        if history["loss"][-1] < best_loss:
            best_loss = history["loss"][-1]
            best_run = i
    return history_file_list[best_run].split("/model/")[0].split("/")[-1]


def save_inf_params(data, output_dir):
    """Save the inferred parameters."""
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    best_run = get_best_run(output_dir)
    model = mdynemo.Model.load(f"{output_dir}/{best_run}/model")

    alpha, beta = model.get_mode_time_courses(data)
    _, stds, corrs = model.get_means_stds_corrs()

    save(f"{inf_params_dir}/alp.pkl", alpha)
    save(f"{inf_params_dir}/bet.pkl", beta)
    save(f"{inf_params_dir}/stds.npy", stds)
    save(f"{inf_params_dir}/corrs.npy", corrs)


def plot_mtc(data, output_dir):
    """Plot the mode time courses."""

    figures_dir = f"{output_dir}/best_run/figures"
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    os.makedirs(figures_dir, exist_ok=True)

    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")

    # Order the modes
    bet_order = get_mode_orders(np.square(stds), corrs)
    beta = [b[:, bet_order] for b in beta]
    corrs = corrs[bet_order]

    norm_alpha = modes.reweight_mtc(alpha, np.square(stds), "covariance")
    norm_beta = modes.reweight_mtc(beta, corrs, "correlation")
    fig, axes = plotting.plot_alpha(
        norm_alpha[0],
        norm_beta[0],
        n_samples=2000,
        sampling_frequency=data.sampling_frequency,
        cmap="tab10",
    )
    axes[0].set_ylabel("Power (re-normalised)", fontsize=14)
    axes[1].set_ylabel("FC (re-normalised)", fontsize=14)
    axes[1].set_xlabel("Time (s)", fontsize=14)
    axes[0].set_title("Re-normalised mode time courses", fontsize=18)
    fig.savefig(f"{figures_dir}/renorm_mtc.png")


def plot_networks(data, output_dir):
    """Plot the networks."""

    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")
    n_modes = stds.shape[0]
    n_corr_modes = corrs.shape[0]

    # Order the modes
    bet_order = get_mode_orders(stds, corrs)
    corrs = corrs[bet_order]

    power.save(
        np.square(stds),
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        filename=f"{figures_dir}/vars.png",
        combined=True,
        titles=[f"M-DyNeMo power mode {i+1}" for i in range(n_modes)],
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
        titles=[f"M-DyNeMo FC mode {i+1}" for i in range(n_corr_modes)],
        filename=f"{figures_dir}/corrs.png",
    )

    corrs_degree = np.sum(corrs - np.eye(corrs.shape[1]), axis=-1)
    power.save(
        corrs_degree,
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        combined=True,
        titles=[f"M-DyNeMo FC degree mode {i+1}" for i in range(n_corr_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"{figures_dir}/corrs_degree.png",
    )


def plot_mode_coupling(data, output_dir):
    """Plot the mode coupling (correlation profile of mode time courses)."""

    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")
    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    n_modes = alpha[0].shape[1]
    n_corr_modes = beta[0].shape[1]

    # Order the modes
    bet_order = get_mode_orders(stds, corrs)
    beta = [b[:, bet_order] for b in beta]

    corr = [np.corrcoef(a, b, rowvar=False) for a, b in zip(alpha, beta)]
    g_corr = np.nanmean(corr, axis=0)

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
    ax[0].set_title("power - power", fontsize=22)
    sns.heatmap(
        g_corr[n_corr_modes:, n_corr_modes:],
        annot=sig_labels[n_corr_modes:, n_corr_modes:],
        fmt="s",
        cmap="coolwarm",
        ax=ax[1],
        vmin=-vmax,
        vmax=vmax,
    )
    ax[1].set_title("FC - FC", fontsize=22)
    sns.heatmap(
        g_corr[:n_modes, n_corr_modes:],
        annot=sig_labels[:n_modes, n_corr_modes:],
        fmt="s",
        cmap="coolwarm",
        ax=ax[2],
        vmin=-vmax,
        vmax=vmax,
    )
    ax[2].set_title("power - FC", fontsize=22)
    for index, a in enumerate(ax):
        a.set_xticklabels(range(1, 5))
        a.set_yticklabels(range(1, 5))
        if index == 2:
            a.set_xlabel("FC mode", fontsize=16)
            a.set_ylabel("power mode", fontsize=16)
        elif index == 1:
            a.set_xlabel("FC mode", fontsize=16)
            a.set_ylabel("FC mode", fontsize=16)
        else:
            a.set_xlabel("power mode", fontsize=16)
            a.set_ylabel("power mode", fontsize=16)
    fig.suptitle(
        "Correlation profile of M-DyNeMo inferred mode time courses", fontsize=24
    )
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/mode_coupling.png")


def _sw_cov(
    data,
    alpha,
    beta,
    window_length,
    step_size,
    shuffle_window_length=None,
    n_jobs=1,
):
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

    if shuffle_window_length is not None:
        # Trim
        for i in range(data.n_sessions):
            n_samples = time_series[i].shape[0]
            n_windows = n_samples // shuffle_window_length
            time_series[i] = time_series[i][: n_windows * shuffle_window_length]
            alpha[i] = alpha[i][: n_windows * shuffle_window_length]
            beta[i] = beta[i][: n_windows * shuffle_window_length]

    sw_cov = connectivity.sliding_window_connectivity(
        time_series,
        window_length=window_length,
        step_size=step_size,
        conn_type="cov",
        n_jobs=n_jobs,
    )
    return sw_cov, alpha, beta


def regress_on_mtc(data, output_dir, window_length, step_size):
    """Regress the power and FC on the sliding window covariance."""

    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")
    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    n_modes = alpha[0].shape[1]
    n_corr_modes = beta[0].shape[1]

    # Order the modes
    bet_order = get_mode_orders(stds, corrs)
    beta = [b[:, bet_order] for b in beta]

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
    )
    sw_a = power.sliding_window_power(
        alpha,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
    )
    sw_b = power.sliding_window_power(
        beta,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
    )

    a_covs = []
    b_covs = []
    for i in range(data.n_sessions):
        a_covs.append(
            regression.linear(
                sw_a[i],
                sw_cov[i],
                fit_intercept=False,
            )
        )
        b_covs.append(
            regression.linear(
                sw_b[i],
                sw_cov[i],
                fit_intercept=False,
            )
        )

    group_a_covs = np.nanmean(a_covs, axis=0)
    group_b_covs = np.nanmean(b_covs, axis=0)
    np.save(f"{inf_params_dir}/a_covs.npy", group_a_covs)
    np.save(f"{inf_params_dir}/b_covs.npy", group_b_covs)

    power.save(
        group_a_covs,
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        combined=True,
        titles=[f"M-DyNeMo mode {i+1}" for i in range(n_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"{figures_dir}/a_vars.png",
    )

    thres_group_a_covs = connectivity.threshold(
        group_a_covs,
        percentile=90,
        absolute_value=True,
    )
    connectivity.save(
        thres_group_a_covs,
        parcellation_file=data.parcellation_file,
        combined=True,
        filename=f"{figures_dir}/a_covs.png",
    )

    power.save(
        group_b_covs,
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        combined=True,
        titles=[f"M-DyNeMo mode {i+1}" for i in range(n_corr_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"{figures_dir}/b_vars.png",
    )

    thres_group_b_covs = connectivity.threshold(
        group_b_covs,
        percentile=90,
        absolute_value=True,
    )
    connectivity.save(
        thres_group_b_covs,
        parcellation_file=data.parcellation_file,
        combined=True,
        filename=f"{figures_dir}/b_covs.png",
    )


def get_mode_orders(stds, corrs):
    """Get the FC mode orders based on cosine similarity.

    Parameters
    ----------
    stds : np.ndarray
        The standard deviations of the modes.
        Shape is (n_modes, n_channels, n_channels) or (n_modes, n_channels).
    corrs : np.ndarray
        The correlations of the modes.
        Shape is (n_modes, n_channels, n_channels).

    Returns
    -------
    order : np.ndarray
        The order of the modes.
    """
    if stds.ndim == 3:
        stds = np.diagonal(stds, axis1=1, axis2=2)

    n_channels = stds.shape[1]

    s = stds - np.mean(stds, axis=0, keepdims=True)
    f = corrs - np.mean(corrs, axis=0, keepdims=True)

    # fill diagonal with 0
    m, n = np.diag_indices(n_channels)
    f[:, m, n] = 0
    f = np.sum(f, axis=-1)

    # match modes
    order = modes.match_vectors(
        s, f, comparison="cosine_similarity", return_order=True
    )[1]

    return order


def get_regressed_power_fc(sw_cov, alpha, beta, window_length, step_size):
    """Get the regressed power and FC.

    Parameters
    ----------
    sw_cov : list of array-like
        The sliding window covariance.
    alpha : list of array-like
        The power mode time courses.
    beta : list of array-like
        The FC mode time courses.
    window_length : int
        The window length.
    step_size : int
        The step size.

    Returns
    -------
    s : array-like
        The regressed power. Shape is (n_modes, n_channels).
    f : array-like
        The regressed FC. Shape is (n_modes, n_channels, n_channels).
    """
    n_sessions = len(sw_cov)

    # Sliding window alpha
    sw_a = power.sliding_window_power(
        alpha,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
    )

    # Sliding window beta
    sw_b = power.sliding_window_power(
        beta,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
    )

    # Regress
    a_stds = []
    for i in range(n_sessions):
        a_stds.append(
            regression.linear(
                sw_a[i],
                np.diagonal(sw_cov[i], axis1=1, axis2=2),
                fit_intercept=False,
            )
        )
    g_a_stds = np.mean(a_stds, axis=0)

    b_fcs = []
    for i in range(n_sessions):
        b_fcs.append(
            regression.linear(
                sw_b[i],
                array_ops.cov2corr(sw_cov[i]) - np.eye(sw_cov[i].shape[1]),
                fit_intercept=False,
            )
        )
    g_b_fcs = np.mean(b_fcs, axis=0)

    # fill diagonal with 0
    m, n = np.diag_indices(g_b_fcs.shape[1])
    g_b_fcs[:, m, n] = 0

    s = g_a_stds - np.mean(g_a_stds, axis=0, keepdims=True)
    f = g_b_fcs - np.mean(g_b_fcs, axis=0, keepdims=True)

    return s, f


def get_regressed_covariance(sw_cov, mtc, window_length, step_size, n_jobs=1):
    """Get the regressed covariance.

    Parameters
    ----------
    sw_cov : list of array-like
        The sliding window covariance.
    mtc : list of array-like
        The mode time courses.
    window_length : int
        The window length.
    step_size : int
        The step size.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1.

    Returns
    -------
    g_covs : array-like
        The regressed covariance. Shape is (n_modes, n_channels, n_channels).
    """
    n_sessions = len(sw_cov)

    # Sliding window
    sw_mtc = power.sliding_window_power(
        mtc,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
        n_jobs=n_jobs,
    )

    # Regress
    covs = []
    for i in range(n_sessions):
        covs.append(
            regression.linear(
                sw_mtc[i],
                sw_cov[i],
                fit_intercept=False,
            )
        )
    g_covs = np.mean(covs, axis=0)

    return g_covs


def similarity_with_dynemo(
    data,
    output_dir,
    dynemo_dir,
    window_length,
    step_size,
    shuffle_window_length,
    n_perm=1000,
    n_jobs=1,
):
    """
    Get the observed cosine similarity between the DyNeMo networks with re-calculated networks
    from M-DyNeMo inferred power and FC mode time courses. Also samples from the null distribution.
    Results are saved in '{output_dir}/best_run/results'.

    Parameters
    ----------
    dynemo_dir : str
        The DyNeMo results directory.
    window_length : int
        The window length.
    step_size : int
        The step size.
    shuffle_window_length : int
        The shuffle window length.
    n_perm : int, optional
        The number of permutations, by default 1000.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1.
    """
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    dynemo_inf_params_dir = f"{dynemo_dir}/best_run/inf_params"
    results_dir = f"{output_dir}/best_run/results"
    os.makedirs(results_dir, exist_ok=True)

    dynemo_covs = load(f"{dynemo_inf_params_dir}/covs.npy")
    dynemo_covs = data.pca_components @ dynemo_covs @ data.pca_components.T

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")
    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    n_modes = alpha[0].shape[1]
    assert n_modes == beta[0].shape[1]
    assert n_modes == dynemo_covs.shape[0]

    bet_order = get_mode_orders(stds, corrs)
    beta = [b[:, bet_order] for b in beta]
    corrs = corrs[bet_order]

    dynemo_order = modes.match_vectors(
        np.diagonal(stds, axis1=1, axis2=2),
        np.diagonal(dynemo_covs, axis1=1, axis2=2),
        comparison="cosine_similarity",
        return_order=True,
    )[1]
    dynemo_covs = dynemo_covs[dynemo_order]

    sw_cov, alpha, beta = _sw_cov(
        data,
        alpha,
        beta,
        window_length,
        step_size,
        shuffle_window_length=shuffle_window_length,
        n_jobs=n_jobs,
    )

    def _cosine_similarity(mtc, ind=None, match_modes=True):
        mtc_copy = [m.copy() for m in mtc]
        # shuffle if ind is not None
        if ind is not None:
            for i in range(data.n_sessions):
                m = mtc_copy[i].reshape(
                    -1, shuffle_window_length, mtc_copy[i].shape[-1]
                )
                m = m[ind[i]]
                mtc_copy[i] = m.reshape(-1, mtc_copy[i].shape[-1])

        g_covs = get_regressed_covariance(sw_cov, mtc_copy, window_length, step_size)
        d_covs = dynemo_covs - np.mean(dynemo_covs, axis=0, keepdims=True)
        g_covs = g_covs - np.mean(g_covs, axis=0, keepdims=True)

        c1 = d_covs.reshape(dynemo_covs.shape[0], -1)
        c2 = g_covs.reshape(g_covs.shape[0], -1)

        # match modes
        if match_modes:
            c1, c2 = modes.match_vectors(c1, c2, comparison="cosine_similarity")

        cos = []
        for i in range(n_modes):
            cos.append(1 - cosine(c1[i], c2[i]))
        return cos

    obs_alp_cosine = _cosine_similarity(alpha, match_modes=False)
    obs_bet_cosine = _cosine_similarity(beta, match_modes=False)

    np.save(f"{results_dir}/obs_alp_cosine.npy", np.array(obs_alp_cosine))
    np.save(f"{results_dir}/obs_bet_cosine.npy", np.array(obs_bet_cosine))

    # null distribution
    n_windows = [m.shape[0] // shuffle_window_length for m in alpha]

    # get the indices for each permutation
    alp_kwargs, bet_kwargs = [], []
    for _ in range(n_perm):
        ind = [np.random.permutation(n_w) for n_w in n_windows]
        alp_kwargs.append({"ind": ind})
        ind = [np.random.permutation(n_w) for n_w in n_windows]
        bet_kwargs.append({"ind": ind})

    if n_jobs == 1:
        null_alp_cosine = []
        for kws in tqdm(alp_kwargs, desc="Permutation"):
            null_alp_cosine.append(_cosine_similarity(alpha, **kws))

        null_bet_cosine = []
        for kws in tqdm(bet_kwargs, desc="Permutation"):
            null_bet_cosine.append(_cosine_similarity(beta, **kws))

    else:
        null_alp_cosine = pqdm(
            alp_kwargs,
            _cosine_similarity,
            desc="Permutation",
            argument_type="kwargs",
            n_jobs=n_jobs,
        )
        null_bet_cosine = pqdm(
            bet_kwargs,
            _cosine_similarity,
            desc="Permutation",
            argument_type="kwargs",
            n_jobs=n_jobs,
        )

    np.save(f"{results_dir}/null_alp_cosine.npy", np.array(null_alp_cosine))
    np.save(f"{results_dir}/null_bet_cosine.npy", np.array(null_bet_cosine))


def null_spatial_map_similarity(
    data,
    output_dir,
    window_length,
    step_size,
    shuffle_window_length,
    n_perm=1000,
    n_jobs=1,
):
    """Get samples from the null distribution for spatial map similarity.
    Results are saved in '{output_dir}/best_run/results'.

    Parameters
    ----------
    window_length : int
        The window length.
    step_size : int
        The step size.
    shuffle_window_length : int
        The shuffle window length.
    n_perm : int, optional
        The number of permutations, by default 1000.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1.
    """
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    results_dir = f"{output_dir}/best_run/results"
    os.makedirs(results_dir, exist_ok=True)

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")
    alpha = load(f"{inf_params_dir}/alp.pkl")
    beta = load(f"{inf_params_dir}/bet.pkl")
    n_modes = alpha[0].shape[1]
    assert n_modes == beta[0].shape[1]

    # Order the modes
    bet_order = get_mode_orders(stds, corrs)
    beta = [b[:, bet_order] for b in beta]

    sw_cov, alpha, beta = _sw_cov(
        data,
        alpha,
        beta,
        window_length,
        step_size,
        shuffle_window_length=shuffle_window_length,
        n_jobs=n_jobs,
    )

    null_spatial_cosine = []
    mtc = [np.concatenate([a, b], axis=1) for a, b in zip(alpha, beta)]

    def _null_spatial_cosine(ind):
        mtc_copy = [m.copy() for m in mtc]
        shuffled_alpha, shuffled_beta = [], []
        for i in range(data.n_sessions):
            m = mtc_copy[i].reshape(-1, shuffle_window_length, mtc_copy[i].shape[-1])
            m = m[ind[i]]
            mtc_copy[i] = m.reshape(-1, mtc_copy[i].shape[-1])
            shuffled_alpha.append(mtc_copy[i][:, : alpha[0].shape[1]])
            shuffled_beta.append(mtc_copy[i][:, alpha[0].shape[1] :])

        s, f = get_regressed_power_fc(
            sw_cov,
            shuffled_alpha,
            shuffled_beta,
            window_length=window_length,
            step_size=step_size,
        )
        f_order = get_mode_orders(s, f)
        f = f[f_order]
        f = np.sum(f, axis=-1)
        cos = []
        for j in range(n_modes):
            cos.append(1 - cosine(s[j], f[j]))
        return cos

    # number of windows for each session
    n_windows = [m.shape[0] // shuffle_window_length for m in mtc]

    # get the indices for each permutation
    kwargs = []
    for _ in range(n_perm):
        ind = [np.random.permutation(n_w) for n_w in n_windows]
        kwargs.append({"ind": ind})

    if n_jobs == 1:
        null_spatial_cosine = []
        for kws in tqdm(kwargs, desc="Permutation"):
            null_spatial_cosine.append(_null_spatial_cosine(**kws))

    else:
        null_spatial_cosine = pqdm(
            kwargs,
            _null_spatial_cosine,
            desc="Permutation",
            argument_type="kwargs",
            n_jobs=n_jobs,
        )

    np.save(f"{results_dir}/null_spatial_cosine.npy", np.array(null_spatial_cosine))


def plot_spatial_map_similarity(data, output_dir):
    """
    Plot the spatial map similarity.
    It is assumed that 'null_spatial_map_similarity' has been run.
    """

    inf_params_dir = f"{output_dir}/best_run/inf_params"
    results_dir = f"{output_dir}/best_run/results"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    stds = load(f"{inf_params_dir}/stds.npy")
    corrs = load(f"{inf_params_dir}/corrs.npy")

    n_modes = stds.shape[0]

    # Order the modes
    bet_order = get_mode_orders(stds, corrs)
    corrs = corrs[bet_order]

    s_diag = np.diagonal(np.square(stds), axis1=1, axis2=2)
    s = s_diag - np.mean(s_diag, axis=0, keepdims=True)
    f = corrs - np.mean(corrs, axis=0, keepdims=True)
    f = np.sum(f, axis=-1)

    spatial_cosine = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            spatial_cosine[i, j] = 1 - cosine(s[i], f[j])

    null_spatial_cosine = np.load(f"{results_dir}/null_spatial_cosine.npy")

    # significant entries
    p = np.mean(
        np.diag(spatial_cosine) < null_spatial_cosine.max(axis=1, keepdims=True),
        axis=0,
    )
    labels = [
        [f"{spatial_cosine[i, j]:.2f}" for j in range(n_modes)] for i in range(n_modes)
    ]
    for i in range(n_modes):
        labels[i][i] += " *" if p[i] < 0.05 and p[i] > 0.01 else ""
        labels[i][i] += " **" if p[i] <= 0.01 and p[i] > 0.001 else ""
        labels[i][i] += " ***" if p[i] <= 0.001 else ""

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        spatial_cosine,
        annot=np.array(labels),
        fmt="s",
        cmap="coolwarm",
        ax=ax,
        annot_kws={"fontsize": 22},
    )
    ax.set_xticklabels(range(1, n_modes + 1), fontsize=24)
    ax.set_yticklabels(range(1, n_modes + 1), fontsize=24)
    ax.set_xlabel("FC modes", fontsize=24)
    ax.set_ylabel("Power modes", fontsize=24)
    ax.set_title("Cosine similarity between spatial maps", fontsize=28)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([])
    fig.savefig(f"{figures_dir}/spatial_map_similarity.png")


def split_half_similarity(
    data,
    output_dir,
    window_length,
    step_size,
    shuffle_window_length,
    n_perm=1000,
    n_jobs=1,
):
    """Get the observed and null samples of split-half similarity.
    Results are saved in '{output_dir}/cosine_similarity'.

    Parameters
    ----------
    window_length : int
        The window length.
    step_size : int
        The step size.
    shuffle_window_length : int
        The shuffle window length.
    n_perm : int, optional
        The number of permutations, by default 1000.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default 1.
    """
    results_dir = f"{output_dir}/cosine_similarity"
    os.makedirs(results_dir, exist_ok=True)

    n_sessions = data.n_sessions
    data_0 = Data(data[: n_sessions // 2])
    data_1 = Data(data[n_sessions // 2 :])

    inf_params_dir_0 = f"{output_dir}/0/best_run/inf_params"
    inf_params_dir_1 = f"{output_dir}/1/best_run/inf_params"

    stds_0 = load(f"{inf_params_dir_0}/stds.npy")
    corrs_0 = load(f"{inf_params_dir_0}/corrs.npy")
    alpha_0 = load(f"{inf_params_dir_0}/alp.pkl")
    beta_0 = load(f"{inf_params_dir_0}/bet.pkl")

    stds_1 = load(f"{inf_params_dir_1}/stds.npy")
    corrs_1 = load(f"{inf_params_dir_1}/corrs.npy")
    alpha_1 = load(f"{inf_params_dir_1}/alp.pkl")
    beta_1 = load(f"{inf_params_dir_1}/bet.pkl")

    # match the modes
    alp_order = modes.match_vectors(
        np.diagonal(stds_0, axis1=1, axis2=2),
        np.diagonal(stds_1, axis1=1, axis2=2),
        comparison="cosine_similarity",
        return_order=True,
    )[1]
    bet_order = modes.match_vectors(
        np.reshape(corrs_0, (corrs_0.shape[0], -1)),
        np.reshape(corrs_1, (corrs_1.shape[0], -1)),
        comparison="cosine_similarity",
        return_order=True,
    )[1]

    alpha_1 = [a[:, alp_order] for a in alpha_1]
    beta_1 = [b[:, bet_order] for b in beta_1]
    stds_1 = stds_1[alp_order]
    corrs_1 = corrs_1[bet_order]

    n_modes = stds_0.shape[0]
    n_channels = stds_0.shape[1]
    obs_power_cosine = []
    obs_fc_cosine = []
    m, n = np.tril_indices(n_channels, -1)
    for i in range(n_modes):
        obs_power_cosine.append(
            1
            - cosine(
                np.diagonal(stds_0[i], axis1=1, axis2=2),
                np.diagonal(stds_1[i], axis1=1, axis2=2),
            )
        )
        obs_fc_cosine.append(1 - cosine(corrs_0[i, m, n], corrs_1[i, m, n]))

    np.save(
        f"{results_dir}/obs_power_cosine.npy",
        np.array(obs_power_cosine),
    )
    np.save(f"{results_dir}/obs_fc_cosine.npy", np.array(obs_fc_cosine))

    # null distribution
    sw_cov_0, alpha_0, beta_0 = _sw_cov(
        data_0,
        alpha_0,
        beta_0,
        window_length,
        step_size,
        shuffle_window_length=shuffle_window_length,
        n_jobs=n_jobs,
    )
    sw_cov_1, alpha_1, beta_1 = _sw_cov(
        data_1,
        alpha_1,
        beta_1,
        window_length,
        step_size,
        shuffle_window_length=shuffle_window_length,
        n_jobs=n_jobs,
    )
    mtc_0 = [np.concatenate([a, b], axis=1) for a, b in zip(alpha_0, beta_0)]
    mtc_1 = [np.concatenate([a, b], axis=1) for a, b in zip(alpha_1, beta_1)]

    def _cosine_similarity(ind_0, ind_1):
        mtc_0_copy = [m.copy() for m in mtc_0]
        mtc_1_copy = [m.copy() for m in mtc_1]
        shuffled_alpha_0, shuffled_beta_0 = [], []
        shuffled_alpha_1, shuffled_beta_1 = [], []

        for i in range(data_0.n_sessions):
            m = mtc_0_copy[i].reshape(
                -1, shuffle_window_length, mtc_0_copy[i].shape[-1]
            )
            m = m[ind_0[i]]
            mtc_0_copy[i] = m.reshape(-1, mtc_0_copy[i].shape[-1])
            shuffled_alpha_0.append(mtc_0_copy[i][:, : alpha_0[0].shape[1]])
            shuffled_beta_0.append(mtc_0_copy[i][:, alpha_0[0].shape[1] :])

        for i in range(data_1.n_sessions):
            m = mtc_1_copy[i].reshape(
                -1, shuffle_window_length, mtc_1_copy[i].shape[-1]
            )
            m = m[ind_1[i]]
            mtc_1_copy[i] = m.reshape(-1, mtc_1_copy[i].shape[-1])
            shuffled_alpha_1.append(mtc_1_copy[i][:, : alpha_1[0].shape[1]])
            shuffled_beta_1.append(mtc_1_copy[i][:, alpha_1[0].shape[1] :])

        s_0, f_0 = get_regressed_power_fc(
            sw_cov_0,
            shuffled_alpha_0,
            shuffled_beta_0,
            window_length=window_length,
            step_size=step_size,
        )
        s_1, f_1 = get_regressed_power_fc(
            sw_cov_1,
            shuffled_alpha_1,
            shuffled_beta_1,
            window_length=window_length,
            step_size=step_size,
        )

        s_0 = s_0 - np.mean(s_0, axis=0, keepdims=True)
        f_0 = (f_0 - np.mean(f_0, axis=0, keepdims=True))[:, m, n]
        s_1 = s_1 - np.mean(s_1, axis=0, keepdims=True)
        f_1 = (f_1 - np.mean(f_1, axis=0, keepdims=True))[:, m, n]

        s_0, s_1 = modes.match_vectors(s_0, s_1, comparison="cosine_similarity")
        f_0, f_1 = modes.match_vectors(f_0, f_1, comparison="cosine_similarity")

        power_cosine, fc_cosine = [], []
        for i in range(n_modes):
            power_cosine.append(1 - cosine(s_0[i], s_1[i]))
            fc_cosine.append(1 - cosine(f_0[i], f_1[i]))

        return power_cosine, fc_cosine

    # number of windows for each session
    n_windows_0 = [m.shape[0] // shuffle_window_length for m in mtc_0]
    n_windows_1 = [m.shape[0] // shuffle_window_length for m in mtc_1]

    kwargs = []
    for _ in range(n_perm):
        ind_0 = [np.random.permutation(n_w) for n_w in n_windows_0]
        ind_1 = [np.random.permutation(n_w) for n_w in n_windows_1]
        kwargs.append({"ind_0": ind_0, "ind_1": ind_1})

    if n_jobs == 1:
        null_power_cosine, null_fc_cosine = [], []
        for kws in tqdm(kwargs, desc="Permutation"):
            power_cosine, fc_cosine = _cosine_similarity(**kws)
            null_power_cosine.append(power_cosine)
            null_fc_cosine.append(fc_cosine)

    else:
        results = pqdm(
            kwargs,
            _cosine_similarity,
            desc="Permutation",
            argument_type="kwargs",
            n_jobs=n_jobs,
        )
        # unpack results
        null_power_cosine, null_fc_cosine = [], []
        for power_cosine, fc_cosine in results:
            null_power_cosine.append(power_cosine)
            null_fc_cosine.append(fc_cosine)

    np.save(
        f"{results_dir}/null_power_cosine.npy",
        np.array(null_power_cosine),
    )
    np.save(
        f"{results_dir}/null_fc_cosine.npy",
        np.array(null_fc_cosine),
    )
