"""Script to train the M-DyNeMo model on the MEGUK-52 dataset."""

from sys import argv

if len(argv) != 2:
    print(f"Please pass the run id, e.g. python {argv[0]}.py 1")
    exit()
id = int(argv[1])

from glob import glob
import os

import pandas as pd

from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops

tf_ops.gpu_growth()


def get_data_df():
    # Helper functions
    def _get_subject(x):
        return os.path.basename(x).split("_")[0].split("-")[1]

    def _get_site(x):
        return _get_subject(x)[:3]

    def _get_task(x):
        return os.path.basename(x).split("_")[1].split("-")[1].split(".")[0]

    def _get_run(x):
        x_split = os.path.basename(x).split("_")
        if len(x_split) == 2:
            return 1
        elif len(x_split) == 3:
            return int(x_split[2].split("-")[1].split(".")[0])

    df = pd.DataFrame()
    data_dir = "/well/woolrich/projects/mrc_meguk/all_sites/npy/src"
    df["data_path"] = sorted(glob(f"{data_dir}/*"))
    df["subject"] = df["data_path"].apply(_get_subject)
    df["site"] = df["data_path"].apply(_get_site)
    df["task"] = df["data_path"].apply(_get_task)
    df["run"] = df["data_path"].apply(_get_run)
    df = df[df["site"] == "not"]
    df = df[df["task"].isin(["resteyesopen", "visuomotor"])]
    return df


def load_data():
    df = get_data_df()
    data = Data(
        df["data_path"].tolist(),
        sampling_frequency=250,
        use_tfrecord=True,
        buffer_size=2000,
        n_jobs=12,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        store_dir=f"tmp/mdynemo/run{id:02d}/",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    )
    data.prepare(
        {
            "amplitude_envelope": {},
            "moving_average": {"n_window": 25},
            "standardize": {},
            "pca": {"n_pca_components": data.n_channels},
        }
    )
    return data


config = f"""
    train_mdynemo:
        config_kwargs:
            n_modes: 4
            learn_means: False
            learn_stds: True
            learn_corrs: True
            learning_rate: 0.001
            batch_size: 256
        init_kwargs:
            n_init: 10
        save_inf_params: False
"""
data = load_data()
run_pipeline(
    config,
    data=data,
    output_dir=f"results/notts_52_mdynemo/run{id:02d}",
)
