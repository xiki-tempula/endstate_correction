#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -pe smp 2
#$ -cwd
#$ -l gpu=1

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate rew

hostname

python sampling.py 1 $1 # first argument = run_id, seconda argument = zinc_id
