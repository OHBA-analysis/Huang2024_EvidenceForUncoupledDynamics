import numpy as np
import pickle
import os

from osl_dynamics.models import mdynemo
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils.misc import load, save
from osl_dynamics.analysis import power, connectivity
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

    best_run_0 = get_best_run(f"{output_dir}/0")
    best_run_1 = get_best_run(f"{output_dir}/1")
    model_0 = mdynemo.Model.load(f"{output_dir}/0/{best_run_0:02d}/model")
    model_1 = mdynemo.Model.load(f"{output_dir}/1/{best_run_1:02d}/model")

    _, stds_0, corrs_0 = model_0.get_means_stds_corrs()
    _, stds_1, corrs_1 = model_1.get_means_stds_corrs()

    save(f"{inf_params_dir}/stds_0.npy", stds_0)
    save(f"{inf_params_dir}/corrs_0.npy", corrs_0)
    save(f"{inf_params_dir}/stds_1.npy", stds_1)
    save(f"{inf_params_dir}/corrs_1.npy", corrs_1)


def save_networks(data, output_dir):
    inf_params_dir = f"{output_dir}/best_run/inf_params"
    figures_dir = f"{output_dir}/best_run/figures"
    os.makedirs(figures_dir, exist_ok=True)

    stds_0 = load(f"{inf_params_dir}/stds_0.npy")
    corrs_0 = load(f"{inf_params_dir}/corrs_0.npy")
    stds_1 = load(f"{inf_params_dir}/stds_1.npy")
    corrs_1 = load(f"{inf_params_dir}/corrs_1.npy")

    n_modes = stds_0.shape[0]
    n_corr_modes = corrs_0.shape[0]

    power.save(
        np.square(stds_0),
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        filename=f"{figures_dir}/vars_0.png",
        combined=True,
        titles=[f"mode {i+1}" for i in range(n_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    )

    thres_corrs_0 = connectivity.threshold(
        corrs_0,
        percentile=90,
        absolute_value=True,
    )
    connectivity.save(
        thres_corrs_0,
        parcellation_file=data.parcellation_file,
        combined=True,
        titles=[f"mode {i+1}" for i in range(n_corr_modes)],
        filename=f"{figures_dir}/corrs_0.png",
    )

    power.save(
        np.square(stds_1),
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        subtract_mean=True,
        show_plots=False,
        filename=f"{figures_dir}/vars_1.png",
        combined=True,
        titles=[f"mode {i+1}" for i in range(n_modes)],
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    )

    thres_corrs_1 = connectivity.threshold(
        corrs_1,
        percentile=90,
        absolute_value=True,
    )
    connectivity.save(
        thres_corrs_1,
        parcellation_file=data.parcellation_file,
        combined=True,
        titles=[f"mode {i+1}" for i in range(n_corr_modes)],
        filename=f"{figures_dir}/corrs_1.png",
    )


config = """
    save_inf_params: {}
    save_networks: {}
"""

run_pipeline(
    config,
    output_dir="results/notts_38_mdynemo_split_half/",
)
