"""
Script for comparing variability of reconstructed time-varying covariances
from M-DyNeMo and DyNeMo on MEGUK-38 resting-state data.
"""

from tqdm.auto import tqdm
import numpy as np

from osl_dynamics.utils.misc import load
from osl_dynamics.inference.metrics import riemannian_distance
from osl_dynamics.utils import plotting

mdynemo_inf_params_dir = "results/notts_38_mdynemo/best_run/inf_params"
dynemo_inf_params_dir = "results/notts_38_dynemo/best_run/inf_params"

# Load the parameters
mdynemo_a = load(f"{mdynemo_inf_params_dir}/alp.pkl")
mdynemo_b = load(f"{mdynemo_inf_params_dir}/bet.pkl")
mdynemo_stds = load(f"{mdynemo_inf_params_dir}/stds.npy")
mdynemo_corrs = load(f"{mdynemo_inf_params_dir}/corrs.npy")

dynemo_a = load(f"{dynemo_inf_params_dir}/alp.pkl")
dynemo_covs = load(f"{dynemo_inf_params_dir}/covs.npy")

# Reconstruct the time-varying covariances
mdynemo_tv_covs = []
for a, g in tqdm(
    zip(mdynemo_a, mdynemo_b),
    total=len(mdynemo_a),
    desc="Reconstructing time-varying covariances",
):
    mdynemo_tv_stds = np.sum(a[..., None, None] * mdynemo_stds[None, ...], axis=1)
    mdynemo_tv_corrs = np.sum(g[..., None, None] * mdynemo_corrs[None, ...], axis=1)
    mdynemo_tv_covs.append(mdynemo_tv_stds @ mdynemo_tv_corrs @ mdynemo_tv_stds)

dynemo_tv_covs = []
for a in tqdm(
    dynemo_a, total=len(dynemo_a), desc="Reconstructing time-varying covariances"
):
    dynemo_tv_covs.append(np.sum(a[..., None, None] * dynemo_covs[None, ...], axis=1))

# Compute the Riemannian distances with mean covariance of each subject
mdynemo_rd = []
for tv_cov in tqdm(
    mdynemo_tv_covs, total=len(mdynemo_tv_covs), desc="Computing Riemannian distances"
):
    mean_cov = np.mean(tv_cov, axis=0)
    rd_ = []
    for cov in tv_cov:
        rd_.append(riemannian_distance(cov, mean_cov))
    mdynemo_rd.append(np.mean(rd_))

dynemo_rd = []
for tv_cov in tqdm(
    dynemo_tv_covs, total=len(dynemo_tv_covs), desc="Computing Riemannian distances"
):
    mean_cov = np.mean(tv_cov, axis=0)
    rd_ = []
    for cov in tv_cov:
        rd_.append(riemannian_distance(cov, mean_cov))
    dynemo_rd.append(np.mean(rd_))

mdynemo_rd = np.array(mdynemo_rd)
dynemo_rd = np.array(dynemo_rd)

# Plot the results
plotting.plot_violin(
    np.array([mdynemo_rd, dynemo_rd]),
    ["M-DyNeMo", "DyNeMo"],
    filename="results/notts_38_tv_covs.png",
)
