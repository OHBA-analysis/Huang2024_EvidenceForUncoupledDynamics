"""
This script is used to run the split-half analysis on the MEGUK dataset.

The script includes the following steps:
1. save the inferred parameters from separate halves.
2. save the networks from separate halves.
3. Permutation test on the similarity of the networks from separate halves.
"""

from glob import glob

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics import run_pipeline

from helper_functions import save_inf_params, plot_networks, split_half_similarity

tf_ops.gpu_growth()

MASK_FILE = "MNI152_T1_8mm_brain.nii.gz"
PARCELLATION_FILE = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
DATA_DIR = "/well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks"

all_data_files = sorted(glob(f"{DATA_DIR}/*"))
data_files_0 = all_data_files[: len(all_data_files) // 2]
data_files_1 = all_data_files[len(all_data_files) // 2 :]


def load_data(inputs, **kwargs):
    data = Data(inputs, **kwargs)
    methods = {
        "filter": {"low_freq": 1, "high_freq": 45},
        "amplitude_envelope": {},
        "moving_average": {"n_window": 25},
        "standardize": {},
        "pca": {"n_pca_components": 38},
    }
    data.prepare(methods)
    return data


data_0 = load_data(
    data_files_0,
    sampling_frequency=250,
    mask_file=MASK_FILE,
    parcellation_file=PARCELLATION_FILE,
)
data_1 = load_data(
    data_files_1,
    sampling_frequency=250,
    mask_file=MASK_FILE,
    parcellation_file=PARCELLATION_FILE,
)


config = """
    save_inf_params: {}
    plot_networks: {}
"""

run_pipeline(
    config,
    output_dir="results/notts_38_mdynemo_split_half/0",
    extra_funcs=[save_inf_params, plot_networks],
    data=data_0,
)
run_pipeline(
    config,
    output_dir="results/notts_38_mdynemo_split_half/1",
    extra_funcs=[save_inf_params, plot_networks],
    data=data_1,
)

data = load_data(
    all_data_files,
    sampling_frequency=250,
    mask_file=MASK_FILE,
    parcellation_file=PARCELLATION_FILE,
)

config_similarity = """
    split_half_similarity:
        window_length: 50
        step_size: 5
        shuffle_window_length: 250
        n_perm: 1000
        n_jobs: 10
"""
run_pipeline(
    config_similarity,
    output_dir="results/notts_38_mdynemo_split_half",
    extra_funcs=[split_half_similarity],
    data=data,
)
