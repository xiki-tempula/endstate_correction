#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -pe smp 2
#$ -cwd
#$ -l h='!(node20|node21|node22|node23|node24)'

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate reweighting

hostname

python sampling.py $1