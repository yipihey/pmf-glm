#!/usr/bin/env bash
# Submit from the repository root after bootstrapping the CUDA environment.
# Adapt the GPU request, account, partition, and memory to the cluster.
#SBATCH --job-name=pmf-glm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=pmf-%j.log
set -euo pipefail
export PMF_FFTW_THREADS=4
exec julia --project=environments/cuda --threads="${SLURM_CPUS_PER_TASK:-8}" \
  scripts/run_pmf.jl examples/pmf-recombination.toml --backend=cuda \
  --output="runs/slurm-${SLURM_JOB_ID:?submit this script with sbatch}"
