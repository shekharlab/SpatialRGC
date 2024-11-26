#!/bin/bash -l
#SBATCH --job-name=PARALLEL_RUN
#SBATCH --account=co_kslab
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=kslab_bigmem3_normal
# Wall clock limit:
#SBATCH --time=24:00:00
#SBATCH --output=PARALLEL_RUN_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --error=PARALLEL_RUN_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kushalnimkar@berkeley.edu
#SBATCH --export=ALL

#Note: ADJUST channels option depending on which run is being done (e.g. Anti-rat for CD31 runs, Celbound3 for Melanopsin runs)
#Also, change absolute path links to scripts
module load python
source activate RGC
python spatial_rgc/imaging_scripts/parallel_stitch.py --zlayers 0 1 2 3 4 5 6 7 --trial_name $2 --stitch_threshold 0.15 --num_processes 32 --subdirectory $1
python spatial_rgc/imaging_scripts/parallel_cell_matrix.py --downsample_ratio 4 --trial_name $2 --num_processes 32 --subdirectory $1
python spatial_rgc/imaging_scripts/add_costains.py --downsample_ratio 4 --trial_name $2 --subdirectory $1 --channels Anti-Rat Cellbound3 --zlayers 0 1 2 3 4 5 6 7
