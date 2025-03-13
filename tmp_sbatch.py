"""Submit slurm jobs for running attacks."""

import subprocess
import sys
from pathlib import Path

# SAMPLE_IDS = tuple(range(208))
# SAMPLE_IDS = tuple(range(50))
SAMPLE_IDS = [31, 39]
NUM_DATA_SHARDS = len(SAMPLE_IDS)
# NUM_DATA_SHARDS = 10
HOURS = 1
# MODEL = "llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00"
MODEL = "llama-7b_TextTextText_None_2024-02-01-00-00-00"
# MODEL = "mistral-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00"
# MODEL = "mistral-7b_TextTextText_None_2024-02-01-00-00-00"

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=prompt-inject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time={hours}:00:00
#SBATCH --output %j-{model}_shard{shard}.log

source $HOME/.bashrc
source activate struq

python test_gcg.py -m models/{model} -a gcg --sample_ids {sample_ids}
"""


def main():
    """Submit slurm jobs."""
    num_jobs = 0
    # Shard behaviors to run in parallel
    for i in range(NUM_DATA_SHARDS):
        num_jobs += 1
        ids = [SAMPLE_IDS[idx] for idx in range(i, len(SAMPLE_IDS), NUM_DATA_SHARDS)]
        script = TEMPLATE.format(
            model=MODEL, sample_ids=" ".join(map(str, ids)), shard=i, hours=HOURS
        )
        output_name = f"{MODEL}_shard{i}"
        script_path = f"_tmp/{output_name}.sh"
        Path("_tmp").mkdir(exist_ok=True, parents=True)
        with open(script_path, "w", encoding="utf-8") as file:
            file.write(script)
        print(f"Submitting job at {script_path}")
        output = subprocess.run(["sbatch", script_path], check=True)
        if output.returncode != 0:
            print("Failed to submit job!")
            sys.exit(1)
    print(f"All {num_jobs} jobs submitted successfully!")


if __name__ == "__main__":
    main()
