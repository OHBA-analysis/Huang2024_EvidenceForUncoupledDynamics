"""Script for training DyNeMo on the Wakeman-Henson dataset."""

from sys import argv

if len(argv) != 2:
    print(f"Please pass the run id, e.g. python {argv[0]}.py 1")
    exit()
id = int(argv[1])

from glob import glob

from osl_dynamics.data import Data
from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops


def load_data(data_dir, store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=16):
    """Load the data."""

    data_paths = sorted(glob(f"{data_dir}/sub*_run*/sflip_parc-raw.fif"))
    data = Data(
        data_paths,
        sampling_frequency=250,
        picks="misc",
        reject_by_annotation="omit",
        use_tfrecord=use_tfrecord,
        buffer_size=buffer_size,
        n_jobs=n_jobs,
        store_dir=store_dir,
    )

    methods = {
        "filter": {"low_freq": 1, "high_freq": 45},
        "amplitude_envelope": {},
        "moving_average": {"n_window": 25},
        "standardize": {},
        "pca": {"n_pca_components": data.n_channels},
    }
    data.prepare(methods)
    return data


config = """
    train_dynemo:
        config_kwargs:
            n_modes: 4
            learn_means: False
            learn_covariances: True
            learning_rate: 0.001
            batch_size: 256
        save_inf_params: False
"""

tf_ops.gpu_growth()

tmp_dir = f"tmp/wh_dynemo/run{id:02d}"
data_dir = "/well/woolrich/projects/wakeman_henson/spring23/src"
training_data = load_data(data_dir, tmp_dir, n_jobs=6)
run_pipeline(
    config,
    output_dir=f"results/wh_dynemo/run{id:02d}",
    data=training_data,
)
