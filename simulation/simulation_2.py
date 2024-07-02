import os
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt
import seaborn as sns

from osl_dynamics import data, simulation, array_ops
from osl_dynamics.inference import tf_ops, modes, metrics
from osl_dynamics.models import dynemo, mdynemo
from osl_dynamics.utils import set_random_seed, plotting

tf_ops.gpu_growth()

# Set random seed
set_random_seed(0)

# Directory to hold results
results_dir = "results/simulation_2"
os.makedirs(results_dir, exist_ok=True)
train_models = True

# simulate data
sim = simulation.HMM_MVN(
    n_samples=25600,
    n_modes=5,
    n_channels=40,
    n_covariances_act=3,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
sim.standardize()
sim_stc = sim.mode_time_course

sim_dynamic_covs = sim.get_instantaneous_covariances()
sim_dynamic_stds = np.array(
    [np.diag(std) for std in array_ops.cov2std(sim_dynamic_covs)]
)
sim_dynamic_corrs = array_ops.cov2corr(sim_dynamic_covs)

sim_covs = sim.covariances
sim_stds = array_ops.cov2std(sim_covs)
sim_corrs = array_ops.cov2corr(sim_covs)
training_data = data.Data(sim.time_series)

if train_models:
    dynemo_config = dynemo.Config(
        n_modes=5,
        n_channels=40,
        sequence_length=200,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=False,
        learn_covariances=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=10,
        n_kl_annealing_epochs=40,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=80,
    )

    dynemo_model = dynemo.Model(dynemo_config)
    dynemo_model.random_subset_initialization(
        training_data, n_init=10, n_epochs=5, take=1, use_tqdm=True
    )
    dynemo_model.fit(training_data, use_tqdm=True)
    dynemo_model.save(f"{results_dir}/model/dynemo")

    # Train M-DyNeMo
    mdynemo_config = mdynemo.Config(
        n_modes=5,
        n_channels=40,
        sequence_length=200,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=False,
        learn_stds=True,
        learn_corrs=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=10,
        n_kl_annealing_epochs=40,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=80,
    )
    mdynemo_model = mdynemo.Model(mdynemo_config)
    mdynemo_model.random_subset_initialization(
        training_data, n_init=10, n_epochs=5, take=1, use_tqdm=True
    )
    mdynemo_model.fit(training_data, use_tqdm=True)
    mdynemo_model.save(f"{results_dir}/model/mdynemo")
else:
    dynemo_model = dynemo.Model.load(f"{results_dir}/model/dynemo")
    mdynemo_model = mdynemo.Model.load(f"{results_dir}/model/mdynemo")


# Helper function to get reconstructed time-varying covariances
def mixing(mtc, mode_maps):
    """
    For calculating the dynamic mode maps.

    Parameters
    ----------
    mtc : List of np.ndarray
        Mode time course for each subject, each has shape (n_samples, n_modes).
    mode_maps : np.ndarray
        Spatial maps. Shape is (n_modes, n_channels) or (n_modes, n_channels, n_channels).

    Returns
    -------
    dynamic_maps : List of np.ndarray
        Mixed spatial maps for each subject.
        Each has shape (n_samples, n_channels) or (n_samples, n_channelsm, n_channels).
    """
    if not isinstance(mtc, list):
        mtc = [mtc]

    n_subjects = len(mtc)
    # Match the dimensions for multiplication
    if mode_maps.ndim == 2:
        mtc = [tc[..., None] for tc in mtc]
    elif mode_maps.ndim == 3:
        mtc = [tc[..., None, None] for tc in mtc]
    else:
        raise ValueError(
            f"ndim for mode_maps must be 2 or 3, but it has ndim={mode_maps.n_dim}."
        )

    mode_maps = mode_maps[None, ...]
    dynamic_maps = []
    for subject in trange(n_subjects, desc="Mixing"):
        dynamic_map = np.sum(np.multiply(mtc[subject], mode_maps), axis=1)
        dynamic_maps.append(dynamic_map)

    return dynamic_maps


# Get the results
_, dynemo_covs = dynemo_model.get_means_covariances()
_, mdynemo_stds, mdynemo_corrs = mdynemo_model.get_means_stds_corrs()

dynemo_alpha = dynemo_model.get_alpha(training_data)
mdynemo_alpha, mdynemo_gamma = mdynemo_model.get_mode_time_courses(training_data)

dynemo_stc = modes.argmax_time_courses(dynemo_alpha)
mdynemo_stc = [
    modes.argmax_time_courses(mdynemo_alpha),
    modes.argmax_time_courses(mdynemo_gamma),
]

dynemo_dynamic_covs = mixing(dynemo_alpha, dynemo_covs)[0]
dynemo_dynamic_corrs = array_ops.cov2corr(dynemo_dynamic_covs)
dynemo_dynamic_stds = np.array(
    [np.diag(std) for std in array_ops.cov2std(dynemo_dynamic_covs)]
)

mdynemo_dynamic_stds = mixing(mdynemo_alpha, mdynemo_stds)[0]
mdynemo_dynamic_corrs = mixing(mdynemo_gamma, mdynemo_corrs)[0]
mdynemo_dynamic_covs = (
    mdynemo_dynamic_stds @ mdynemo_dynamic_corrs @ mdynemo_dynamic_stds
)

# Match the modes
_, dynemo_order_alpha = modes.match_modes(sim_stc, dynemo_stc, return_order=True)
dynemo_stc = dynemo_stc[:, dynemo_order_alpha]
dynemo_stds = array_ops.cov2std(dynemo_covs)[dynemo_order_alpha]
dynemo_corrs = array_ops.cov2corr(dynemo_covs)[dynemo_order_alpha]

_, mdynemo_order_alpha = modes.match_modes(sim_stc, mdynemo_stc[0], return_order=True)
_, mdynemo_order_gamma = modes.match_modes(sim_stc, mdynemo_stc[1], return_order=True)
mdynemo_stc = [
    mdynemo_stc[0][:, mdynemo_order_alpha],
    mdynemo_stc[1][:, mdynemo_order_gamma],
]

mdynemo_stds = mdynemo_stds[mdynemo_order_alpha]
mdynemo_corrs = mdynemo_corrs[mdynemo_order_gamma]

# plot state time courses
fig, ax = plotting.plot_alpha(
    sim_stc,
    dynemo_stc,
    mdynemo_stc[0],
    mdynemo_stc[1],
    n_samples=500,
    cmap="tab10",
)
ax[0].set_title("Mode time courses", fontsize=22)
ax[0].set_ylabel("Simulated", fontsize=16)
ax[1].set_ylabel("DyNeMo", fontsize=16)
ax[2].set_ylabel("M-DyNeMo " + r"$\alpha$", fontsize=16)
ax[3].set_ylabel("M-DyNeMo " + r"$\beta$", fontsize=16)
ax[3].set_xlabel("Samples", fontsize=16)
fig.savefig(f"{results_dir}/mode_time_courses.png", dpi=300)

# Riemannian distance from the ground truth
mdynemo_dynamic_covs_rd = np.zeros((5000,))
dynemo_dynamic_covs_rd = np.zeros((5000,))
for n in trange(5000, desc="Computing Riemannian distances"):
    mdynemo_dynamic_covs_rd[n] = metrics.riemannian_distance(
        sim_dynamic_covs[n], mdynemo_dynamic_covs[n]
    )
    dynemo_dynamic_covs_rd[n] = metrics.riemannian_distance(
        sim_dynamic_covs[n], dynemo_dynamic_covs[n]
    )

fig, ax = plotting.plot_line(
    [range(5000), range(5000)],
    [mdynemo_dynamic_covs_rd[:5000], dynemo_dynamic_covs_rd[:5000]],
    labels=["M-DyNeMo", "DyNeMo"],
    title="Riemannian distance from ground truth.",
)
ax.set_xlabel("Samples", fontsize=14)
ax.set_ylabel("Riemannian distance", fontsize=14)
ax.set_title("Riemannian distance from ground truth", fontsize=18)
fig.savefig(f"{results_dir}/riemannian_distance.png", dpi=300)

training_data.delete_dir()
