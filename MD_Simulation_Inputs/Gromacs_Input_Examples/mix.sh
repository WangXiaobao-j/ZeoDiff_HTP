#!/bin/bash
#SBATCH --job-name=EWO               # Job name
#SBATCH --partition=chem             # Partition
#SBATCH --ntasks-per-node=8          # Number of tasks per node
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --account=jxy                # User group
#SBATCH --error=%j.err
#SBATCH --output=%j.out

#----------------------------
# Setup working directory and node list
#----------------------------
CURDIR=`pwd`
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID

NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES; do
    echo "$i:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_ID
done

echo $SLURM_NPROCS
echo $SLURM_GPUS

#----------------------------
# Load software environments
#----------------------------

# Intel oneAPI 2020
source /public/intel/oneapi/setvars.sh

# GROMACS 2023 (modify path/version if needed)
export PATH=/hpc/home/jxy/chem_wangxb/wxb/soft/gromacs-2023/bin:$PATH

# PLUMED 2.9
export PATH=/hpc/home/jxy/chem_wangxb/share_group_folder_jxy/soft/plumed2.9/bin/:$PATH
export LD_LIBRARY_PATH=/hpc/home/jxy/share_group_folder_jxy/soft/plumed2.9/lib/:$LD_LIBRARY_PATH
export PLUMED_KERNEL=/hpc/home/jxy/share_group_folder_jxy/soft/plumed2.9/lib/libplumedKernel.so:$PLUMED_KERNEL

# GCC 9.3
export GCCHOME=/hpc/home/jxy/chem_wangxb/share_group_folder_jxy/soft/gcc-9.3.0/gcc-9.3.0/
export PATH=$GCCHOME/bin:$PATH
export LD_LIBRARY_PATH=$GCCHOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GCCHOME/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$GCCHOME/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$GCCHOME/include/:$CPLUS_INCLUDE_PATH

#----------------------------
# Set executable path
#----------------------------
cd $CURDIR
EXCE=/hpc/home/jxy/chem_wangxb/share_group_folder_jxy/soft/gromacs22_cuda_avx512_plumed2.9

#----------------------------
# Step 1: Energy Minimization (EM)
#----------------------------
if [ ! -e em.tpr ]; then
    gmx grompp -f em.mdp -c mix.gro -p topol.top -n index.ndx -o em.tpr -maxwarn 5
    gmx mdrun -v -deffnm em
fi

#----------------------------
# Step 2: Equilibration (Eq)
#----------------------------
if [ ! -e eq.tpr ]; then
    gmx grompp -f eq.mdp -c em.gro -p topol.top -n index.ndx -o eq.tpr -maxwarn 5
    gmx mdrun -v -deffnm eq
fi

#----------------------------
# Step 3: Production MD
#----------------------------
if [ ! -e md.tpr ]; then
    gmx grompp -f md.mdp -c eq.gro -p topol.top -n index.ndx -o md.tpr -maxwarn 5
    gmx mdrun -v -deffnm md
fi

#----------------------------
# Step 4: Fix periodic boundary for trajectory
#----------------------------
if [ ! -e fixed.xtc ]; then
    echo -e "0\n" | gmx trjconv -f md.xtc -s md.tpr -pbc mol -o fixed.xtc
fi

#----------------------------
# Step 5: Compute Mean Squared Displacement (MSD)
#----------------------------
if [ ! -e msd.xvg ]; then
    echo -e "3\n" | gmx msd -f md.xtc -s md.tpr -o msd.xvg
fi

# MSD for each direction (x, y, z)
if [ ! -e msd_x.xvg ]; then
    echo -e "3\n" | gmx msd -f md.xtc -s md.tpr -type x -o msd_x.xvg
fi

if [ ! -e msd_y.xvg ]; then
    echo -e "3\n" | gmx msd -f md.xtc -s md.tpr -type y -o msd_y.xvg
fi

if [ ! -e msd_z.xvg ]; then
    echo -e "3\n" | gmx msd -f md.xtc -s md.tpr -type z -o msd_z.xvg
fi
