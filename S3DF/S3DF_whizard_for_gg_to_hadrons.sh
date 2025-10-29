#!/bin/bash
#
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --qos=preemptable
#SBATCH --time=5:00:00
#SBATCH --job-name=whizard_for_gg_to_hadrons
#SBATCH --output=whizard_for_gg_to_hadrons.%A_%a.out
#SBATCH --error=whizard_for_gg_to_hadrons.%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --array=0-0

# Seeds to run, space-separated. Override by exporting SEED_LIST in the environment.
SEED_LIST=${SEED_LIST:-"365710 469163 777001 264301 7598420 1321060 225041 505823 459273 012479"}

# Working directory and input template. Override to target a different workspace or SIN file.
export WORK_DIR=${WORK_DIR:-/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS1}
export SIN_TEMPLATE=${SIN_TEMPLATE:-circe_AA_dd_gg_had_ps_C3_250_PS1.gg.sin}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_CONT_CMDOPTS="-B /sdf,/fs"
cd /fs/ddn/sdf/group/atlas/d/dntounis/
export SEED_LIST
# Resolve this task's seed and export it for the payload
SEED_LIST_ARRAY=($SEED_LIST)
export THIS_SEED=${SEED_LIST_ARRAY[$SLURM_ARRAY_TASK_ID]}
export ALRB_CONT_RUNPAYLOAD="source /fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS1/S3DF/payload_whizard_for_gg_to_hadrons.sh ${THIS_SEED}"
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh -c el9 -m /sdf,/fs --pwd $PWD