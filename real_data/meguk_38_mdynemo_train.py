"""Script to train M-DyNeMo on MEGUK-38 resting-state data."""

from sys import argv

if len(argv) != 2:
    print(f"Please pass the run id, e.g. python {argv[0]}.py 1")
    exit()
id = int(argv[1])

from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops

config = f"""
    load_data:
        inputs: /well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks
        kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 8
            use_tfrecord: True
            buffer_size: 2000
            store_dir: tmp/notts_38_mdynemo/run{id:02d}
        prepare:
            filter: {{low_freq: 1, high_freq: 45}}
            amplitude_envelope: {{}}
            moving_average: {{n_window: 101}}
            standardize: {{}}
            pca: {{n_pca_components: 38}}
    train_mdynemo:
        config_kwargs:
            n_modes: 4
            learn_means: False
            learn_stds: True
            learn_corrs: True
            learning_rate: 0.001
            batch_size: 256
            n_epochs: 80
            n_kl_annealing_epochs: 40
        init_kwargs:
            n_init: 10
        save_inf_params: False
"""
tf_ops.gpu_growth()
run_pipeline(
    config,
    output_dir=f"results/notts_38_mdynemo/run{id:02d}",
)
