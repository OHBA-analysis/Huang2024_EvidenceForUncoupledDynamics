"""
Script for analysing and plotting results from running M-DyNeMo on MEGUK-38 resting-state data.

This script includes the following steps:
1. save the inferred parameters.
2. plot the mode time courses.
3. plot the networks.
4. plot the mode coupling (correlation profile of the mode time courses).
5. Re-calculate networks based on the inferred power and FC mode time courses.
6. Permutation test on the spatial map similarity between power and FC networks.
7. Permutation test on network similarity with DyNeMo.
"""

from osl_dynamics.inference import tf_ops
from osl_dynamics import run_pipeline

from helper_functions import (
    save_inf_params,
    plot_mtc,
    plot_networks,
    plot_mode_coupling,
    regress_on_mtc,
    null_spatial_map_similarity,
    plot_spatial_map_similarity,
    similarity_with_dynemo,
)

tf_ops.gpu_growth()


config = """
    load_data:
        inputs: /well/woolrich/projects/toolbox_paper/ctf_rest/training_data/networks
        kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 8
            use_tfrecord: True
            buffer_size: 2000
            store_dir: tmp/notts_38_mdynemo
        prepare:
            filter: {low_freq: 1, high_freq: 45}
            amplitude_envelope: {}
            moving_average: {n_window: 25}
            standardize: {}
            pca: {n_pca_components: 38}
    save_inf_params: {}
    plot_mtc: {}
    plot_networks: {}
    plot_mode_coupling: {}
    regress_on_mtc:
        window_length: 50
        step_size: 5
    null_spatial_map_similarity:
        window_length: 50
        step_size: 5
        shuffle_window_length: 250
        n_perm: 1000
        n_jobs: 10
    plot_spatial_map_similarity: {}
    similarity_with_dynemo:
        dynemo_dir: results/notts_38_dynemo
        window_length: 50
        step_size: 5
        shuffle_window_length: 250
        n_perm: 1000
        n_jobs: 10
"""

run_pipeline(
    config,
    output_dir=f"results/notts_38_mdynemo",
    extra_funcs=[
        save_inf_params,
        plot_mtc,
        plot_networks,
        plot_mode_coupling,
        regress_on_mtc,
        null_spatial_map_similarity,
        plot_spatial_map_similarity,
        similarity_with_dynemo,
    ],
)
