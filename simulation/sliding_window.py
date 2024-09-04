"""Script for running sliding window approach on simulated data."""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from osl_dynamics import data, simulation, array_ops
from osl_dynamics.analysis import connectivity
from osl_dynamics.inference import modes, metrics
from osl_dynamics.utils import set_random_seed, plotting

set_random_seed(0)

results_dir = "results/sliding_window"
os.makedirs(results_dir, exist_ok=True)

# simulate data
sim = simulation.MDyn_HMM_MVN(
    n_samples=51200,
    n_modes=5,
    n_channels=40,
    n_covariances_act=3,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
sim.standardize()
data = data.Data(sim.time_series)
sim_stc = sim.mode_time_course

window_length_list = [4, 6, 8, 10, 12, 14, 16]
train = False
if train:
    dice_std, dice_corr = [], []
    for window_length in window_length_list:
        tv_covs = connectivity.sliding_window_connectivity(
            data.time_series(),
            window_length=window_length,
            step_size=1,
            conn_type="cov",
        )
        tv_std = array_ops.cov2std(tv_covs)
        tv_corr = array_ops.cov2corr(tv_covs)
        flattened_tv_corr = tv_corr.reshape(tv_corr.shape[0], -1)

        std_kmeans = KMeans(
            n_clusters=5, n_init="auto", init="k-means++", random_state=0
        ).fit(tv_std)
        corr_kmeans = KMeans(
            n_clusters=5, n_init="auto", init="k-means++", random_state=0
        ).fit(flattened_tv_corr)

        std_labels = std_kmeans.labels_
        corr_labels = corr_kmeans.labels_

        std_stc = array_ops.get_one_hot(std_labels, 5)
        corr_stc = array_ops.get_one_hot(corr_labels, 5)

        # match modes
        std_stc = modes.match_modes(
            sim_stc[0][window_length // 2 : -window_length // 2], std_stc
        )[1]
        corr_stc = modes.match_modes(
            sim_stc[1][window_length // 2 : -window_length // 2], corr_stc
        )[1]

        dice_std.append(
            metrics.dice_coefficient(
                sim_stc[0][window_length // 2 : -window_length // 2], std_stc
            )
        )
        dice_corr.append(
            metrics.dice_coefficient(
                sim_stc[1][window_length // 2 : -window_length // 2], corr_stc
            )
        )
    dice_std = np.array(dice_std)
    dice_corr = np.array(dice_corr)
    np.save(f"{results_dir}/dice_std.npy", dice_std)
    np.save(f"{results_dir}/dice_corr.npy", dice_corr)
else:
    dice_std = np.load(f"{results_dir}/dice_std.npy")
    dice_corr = np.load(f"{results_dir}/dice_corr.npy")

plotting.plot_line(
    [window_length_list] * 2,
    [dice_std, dice_corr],
    labels=["Std", "Corr"],
    x_label="Window Length",
    y_label="Dice Coefficient",
    title="Dice Coefficient vs Window Length",
    filename=f"{results_dir}/dice_vs_window_length.png",
)

best_window_length = window_length_list[np.argmax(dice_std + dice_corr)]
print(f"Best window length: {best_window_length}")

tv_covs = connectivity.sliding_window_connectivity(
    data.time_series(),
    window_length=best_window_length,
    step_size=1,
    conn_type="cov",
)
tv_std = array_ops.cov2std(tv_covs)
tv_corr = array_ops.cov2corr(tv_covs)
flattened_tv_corr = tv_corr.reshape(tv_corr.shape[0], -1)

std_kmeans = KMeans(n_clusters=5, n_init="auto", init="k-means++", random_state=0).fit(
    tv_std
)
corr_kmeans = KMeans(n_clusters=5, n_init="auto", init="k-means++", random_state=0).fit(
    flattened_tv_corr
)

std_labels = std_kmeans.labels_
corr_labels = corr_kmeans.labels_

std_stc = array_ops.get_one_hot(std_labels, 5)
corr_stc = array_ops.get_one_hot(corr_labels, 5)

# match modes
std_stc = modes.match_modes(
    sim_stc[0][best_window_length // 2 : -best_window_length // 2], std_stc
)[1]
corr_stc = modes.match_modes(
    sim_stc[1][best_window_length // 2 : -best_window_length // 2], corr_stc
)[1]

fig, axes = plt.subplots(2, 2, figsize=(12, 5))
plotting.plot_alpha(
    sim_stc[0][best_window_length // 2 : -best_window_length // 2][700:1200],
    std_stc[700:1200],
    n_samples=1000,
    axes=axes[:, 0],
    y_labels=["Simulated", "SW Inferred"],
    title="Standard Deviation",
    cmap="tab10",
)
plotting.plot_alpha(
    sim_stc[1][best_window_length // 2 : -best_window_length // 2][700:1200],
    corr_stc[700:1200],
    n_samples=1000,
    axes=axes[:, 1],
    y_labels=["Simulated", "SW Inferred"],
    title="Correlation",
    cmap="tab10",
)
fig.savefig(f"{results_dir}/mode_time_courses.png")

data.delete_dir()
