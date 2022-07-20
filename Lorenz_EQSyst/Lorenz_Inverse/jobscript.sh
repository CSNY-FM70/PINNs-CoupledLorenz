#!/usr/local_rwth/bin/zsh

# declare the merged STDOUT/STDERR file
#SBATCH --output=output.txt

module load DEVELOP
module load python
pip3 install tensorflow
pip3 install deepxde

python3 lorenz.py
