#!/bin/bash
#SBATCH --job-name=RASPA                 # Job name
#SBATCH --partition=chem                  # Partition/queue name
#SBATCH --ntasks-per-node=1               # Number of tasks per node
#SBATCH --account=jxy                     # User account/group
#SBATCH --error=%j.err                    # Standard error file
#SBATCH --output=%j.out                   # Standard output file

# -----------------------------
# Set working directory
# -----------------------------
CURDIR=$(pwd)
rm -f $CURDIR/nodelist.$SLURM_JOB_ID

# Generate node list
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
for NODE in $NODES; do
    echo "$NODE:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_ID
done

# Print total number of processes and GPUs
echo "Total processes: $SLURM_NPROCS"
echo "GPUs: $SLURM_GPUS"

# -----------------------------
# Load environment modules / software paths
# -----------------------------

# Intel oneAPI 2020
source /public/intel/oneapi/setvars.sh

# GROMACS 2023
export PATH=/hpc/home/jxy/chem_wangxb/wxb/soft/gromacs-2023/bin:$PATH

# PLUMED 2.9
export PATH=/hpc/home/jxy/chem_wangxb/share_group_folder_jxy/soft/plumed2.9/bin:$PATH
export LD_LIBRARY_PATH=/hpc/home/jxy/share_group_folder_jxy/soft/plumed2.9/lib:$LD_LIBRARY_PATH
export PLUMED_KERNEL=/hpc/home/jxy/share_group_folder_jxy/soft/plumed2.9/lib/libplumedKernel.so:$PLUMED_KERNEL

# GCC 9.3
export GCCHOME=/hpc/home/jxy/chem_wangxb/share_group_folder_jxy/soft/gcc-9.3.0/gcc-9.3.0/
export PATH=$GCCHOME/bin:$PATH
export LD_LIBRARY_PATH=$GCCHOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GCCHOME/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$GCCHOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$GCCHOME/include:$CPLUS_INCLUDE_PATH

# RASPA 2.0
export RASPA_DIR=/hpc/home/jxy/share_group_folder_jxy/soft/raspa
export PATH=$RASPA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$RASPA_DIR/lib:$LD_LIBRARY_PATH

# GROMACS executable (modify as needed)
EXE=/hpc/home/jxy/chem_wangxb/share_group_folder_jxy/soft/gromacs22_cuda_avx512_plumed2.9

# -----------------------------
# Run RASPA simulation
# -----------------------------
cd $CURDIR

# Check if 'Movies' folder exists; run simulation if it does not
if [ ! -d "Movies" ]; then
    simulate simulation.input
fi
