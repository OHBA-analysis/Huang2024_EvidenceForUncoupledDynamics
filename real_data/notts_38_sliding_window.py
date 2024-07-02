import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from osl_dynamics.data import Data
from osl_dynamics.analysis import connectivity, power
from osl_dynamics import array_ops
from osl_dynamics.utils import plotting, set_random_seed

set_random_seed(42)
figures_dir = "results/notts_38_sw"

# Load and prepare data
data = Data(
    "/well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks",
    sampling_frequency=250,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
)
methods = {
    "filter": {"low_freq": 1, "high_freq": 45, "use_raw": True},
    "amplitude_envelope": {},
    "moving_average": {"n_window": 25},
    "standardize": {},
    "pca": {"n_pca_components": data.n_channels},
}
data.prepare(methods)

# Calculate sliding window covariances
tv_covs = connectivity.sliding_window_connectivity(
    data.time_series(),
    window_length=500,
    step_size=10,
    conn_type="cov",
    n_jobs=12,
)

# Get the standard deviations and correlations
tv_std = [array_ops.cov2std(cov) for cov in tv_covs]
tv_corr = [array_ops.cov2corr(cov) for cov in tv_covs]
flattened_tv_corr = [corr.reshape(corr.shape[0], -1) for corr in tv_corr]

# KMeans clustering
std_kmeans = KMeans(
    n_clusters=4,
    n_init="auto",
    init="k-means++",
    random_state=0,
).fit(np.concatenate(tv_std, axis=0))
corr_kmeans = KMeans(
    n_clusters=4,
    n_init="auto",
    init="k-means++",
    random_state=0,
).fit(np.concatenate(flattened_tv_corr, axis=0))

# Get the cluster labels
std_kmeans_stc = [
    array_ops.get_one_hot(std_kmeans.predict(std), n_states=4) for std in tv_std
]
corr_kmeans_stc = [
    array_ops.get_one_hot(corr_kmeans.predict(f_corr), n_states=4)
    for f_corr in flattened_tv_corr
]

# Plot state time course
fig, ax = plotting.plot_alpha(
    std_kmeans_stc[0],
    corr_kmeans_stc[0],
    n_samples=5000,
    cmap="tab10",
    sampling_frequency=250,
)
ax[0].set_title("State time courses with KMeans clustering", fontsize=22)
ax[0].set_ylabel("TV Variance", fontsize=18)
ax[1].set_ylabel("TV Correlation", fontsize=18)
ax[1].set_xlabel("Time (s)", fontsize=18)
fig.savefig(f"{figures_dir}/sw_stc.png", dpi=300)

# Power maps
std_centroids = std_kmeans.cluster_centers_
std_centroids = np.array([np.diag(c) for c in std_centroids])
std_centroids = data.pca_components @ std_centroids @ data.pca_components.T
power.save(
    std_centroids,
    mask_file=data.mask_file,
    parcellation_file=data.parcellation_file,
    subtract_mean=True,
    show_plots=False,
    filename=f"{figures_dir}/sw_var.png",
    combined=True,
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True, "vmax": 0.25},
    titles=["State 1", "State 2", "State 3", "State 4"],
)

# FC maps
corr_centroids = corr_kmeans.cluster_centers_.reshape(
    4, data.n_channels, data.n_channels
)
corr_centroids = array_ops.cov2corr(corr_centroids)
corr_centroids = data.pca_components @ corr_centroids @ data.pca_components.T
thres_corr_centroids = connectivity.threshold(
    corr_centroids, 90, absolute_value=True, subtract_mean=True
)
connectivity.save(
    thres_corr_centroids,
    parcellation_file=data.parcellation_file,
    filename=f"{figures_dir}/sw_corr.png",
    combined=True,
    titles=["State 1", "State 2", "State 3", "State 4"],
)

# Correlation between state time courses
stc_corr = [
    np.corrcoef(
        stc1,
        stc2,
        rowvar=False,
    )
    for stc1, stc2 in zip(std_kmeans_stc, corr_kmeans_stc)
]
stc_corr = np.array(stc_corr)

fig, ax = plt.subplots()
sns.heatmap(
    np.nanmean(stc_corr, axis=0)[:4, 4:],
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    ax=ax,
)
ax.set_title("Correlation between state time courses", fontsize=16)
ax.set_xlabel("TV corr state", fontsize=16)
ax.set_ylabel("TV std state", fontsize=16)
ax.set_xticklabels(range(1, 5), fontsize=14)
ax.set_yticklabels(range(1, 5), fontsize=14)
fig.savefig(f"{figures_dir}/sw_stc_corr.png", dpi=300)

data.delete_dir()