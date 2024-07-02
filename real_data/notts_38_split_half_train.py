from sys import argv

if len(argv) != 3:
    print(f"Please pass the run and half id, e.g. python {argv[0]}.py 1 0")
    exit()

id = int(argv[1])
half_id = int(argv[2])

from glob import glob

from osl_dynamics.inference import tf_ops
from osl_dynamics.data import Data
from osl_dynamics import run_pipeline


def load_data(half_id, **kwargs):
    data_dir = "/well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks"
    all_data_files = sorted(glob(f"{data_dir}/*"))
    if not half_id:
        data_files = all_data_files[: len(all_data_files) // 2]
    else:
        data_files = all_data_files[len(all_data_files) // 2 :]

    data = Data(data_files, **kwargs)
    methods = {
        "filter": {"low_freq": 1, "high_freq": 45},
        "amplitude_envelope": {},
        "moving_average": {"n_window": 25},
        "standardize": {},
        "pca": {"n_pca_components": 38},
    }
    data.prepare(methods)
    return data


config = """
    train_mdynemo:
        config_kwargs:
            n_modes: 4
            learn_means: False
            learn_stds: True
            learn_corrs: True
            learning_rate: 0.001
            batch_size: 256
            n_epochs: 80
            lr_decay: 0.05
            n_kl_annealing_epochs: 40
        init_kwargs:
            n_init: 10
        save_inf_params: False
"""

tf_ops.gpu_growth()

data = load_data(
    half_id,
    sampling_frequency=250,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    n_jobs=8,
    use_tfrecord=True,
    buffer_size=2000,
    store_dir=f"tmp/notts_38_mdynemo_split_half/{half_id}/run{id:02d}",
)
run_pipeline(
    config,
    output_dir=f"results/notts_38_mdynemo_split_half/{half_id}/run{id:02d}",
    data=data,
)
