#!/bin/zsh

# =========== Setting up the cluster job parameters =========== #

#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --job-name=OptError
#SBATCH --mem-per-cpu=4096M
#SBATCH --output=ErrorOpt_%J.txt

#SBATCH --mail-user=miguel.cerdeira@rwth-aachen.de
#SBATCH --mail-type=END

# ============== Loading the necessary Modules ================ #
module unload intelmpi
module switch intel gcc
module load python
module load cuda/11.2
module load cudnn
#pip3 install --<user> tensorflow
#pip3 install --<user> deepxde

# ================ Running the MCMC simulations =============== #

python3 OptErr_1.py
