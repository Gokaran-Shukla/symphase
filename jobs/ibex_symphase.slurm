#!/bin/bash
# =============================================================================
#  SymPhase on one Ibex node (all cores used by the process backend)
#    sbatch ibex_symphase.slurm                      # uses symphase.in in this folder
#    sbatch ibex_symphase.slurm Ti Ni Sn write_vasp=yes
#  NOTE: `symphase run` downloads data (OQMD / Materials Project). Ibex compute
#  nodes may have no internet: run it once on the login node (fast, cached in
#  <outdir>/data/) and use this job for large hull/screening work afterwards.
# =============================================================================
#SBATCH --job-name=symphase
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=symphase_%j.out
#SBATCH --error=symphase_%j.err

set +e
source /home/shuklagn/anaconda3/etc/profile.d/conda.sh
conda activate symphase            # conda create -n symphase python=3.11 && pip install symphase/
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one BLAS thread per worker
export SYMPHASE_NPROC=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg

cd "$SLURM_SUBMIT_DIR"
echo "SymPhase job $SLURM_JOB_ID on $SLURM_JOB_NODELIST with $SLURM_CPUS_PER_TASK cores"
if [ $# -gt 0 ]; then
    symphase run "$@" nproc=$SLURM_CPUS_PER_TASK
else
    symphase run symphase.in nproc=$SLURM_CPUS_PER_TASK
fi
# other examples:
#   symphase hull --scan calcs/ -n $SLURM_CPUS_PER_TASK --plot hull.png --csv-out hull.csv
#   symphase hull --csv refs.csv --screen candidates.csv -n $SLURM_CPUS_PER_TASK
