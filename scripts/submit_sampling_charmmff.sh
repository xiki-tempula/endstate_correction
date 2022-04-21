#$ -S /bin/bash
#$ -M sara.tkaczyk@gmail.com
#$ -m e
#$ -j y
#$ -p -100
#$ -o /data/shared/projects/SGE_LOG/
#$ -pe smp 2
#$ -cwd
#$ -l h='!(node20|node21|node22|node23|node24)'

. /data/shared/projects/anaconda_envs/stkaczyk/etc/profile.d/conda.sh
conda activate rew

hostname

python sampling_charmmff.py $1