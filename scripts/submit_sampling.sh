#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -pe smp 4
#$ -cwd
#$ -l h='!(node17|node18|node19|node20|node21|node22|node23|node24)'

. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
conda activate reweighting

python sampling.py