"""Script for submitting training jobs to the BMRC cluster."""

import os


def write_and_submit_job_script(run, script):
    with open("job.sh", "w") as file:
        name = f"{script}-{run}"
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH -J {name}\n")
        file.write(f"#SBATCH -o outputs/{name}.out\n")
        file.write("#SBATCH -p gpu_short\n")
        file.write(f"#SBATCH --gres gpu:1 --constraint 'a100|v100|rtx8000'\n")
        file.write("source activate osld\n")
        file.write(f"python {script} {run}\n")

    os.system("sbatch job.sh")
    os.system("rm job.sh")


os.makedirs("outputs", exist_ok=True)

for run in range(10):
    write_and_submit_job_script(run, "meguk_38_dynemo_train.py")
    write_and_submit_job_script(run, "meguk_38_mdynemo_train.py")
    write_and_submit_job_script(run, "meguk_38_split_half_train.py")
    write_and_submit_job_script(run, "meguk_52_mdynemo_train.py")
    write_and_submit_job_script(run, "wh_dynemo_train.py")
    write_and_submit_job_script(run, "wh_mdynemo_train.py")
