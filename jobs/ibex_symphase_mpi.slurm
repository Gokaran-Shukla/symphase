#!/bin/bash
# =============================================================================
#  SymPhase over several Ibex nodes with MPI (mpi4py) — for very large
#  screenings / VASP-folder scans.  Every rank builds the hull once; entries and
#  candidates are distributed round-robin; rank 0 writes all output.
#    sbatch ibex_symphase_mpi.slurm
#  needs:  pip install mpi4py   (built against the loaded MPI)
# =============================================================================
#SBATCH --job-name=symphase-mpi
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=symphase_mpi_%j.out
#SBATCH --error=symphase_mpi_%j.err

set +e
source /home/shuklagn/intel/oneapi/setvars.sh > /dev/null 2>&1     # Intel MPI
source /home/shuklagn/anaconda3/etc/profile.d/conda.sh
conda activate symphase
set -e
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR"
echo "SymPhase MPI job $SLURM_JOB_ID: $SLURM_NTASKS ranks on $SLURM_JOB_NODELIST"

srun -n $SLURM_NTASKS python -m symphase hull \
     --csv references.csv \
     --screen candidates.csv --screen-out screening.csv \
     --plot hull.png --csv-out hull_table.csv
# or:  srun -n $SLURM_NTASKS python -m symphase hull --scan calcs/ --plot hull.png
