source /cvmfs/sw.hsf.org/key4hep/setup.sh
cd ${WORK_DIR:-/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS1}

# Resolve seed: prefer CLI arg, then THIS_SEED, then array
CLI_SEED="$1"
if [ -n "${CLI_SEED}" ]; then
  SEED=${CLI_SEED}
elif [ -n "${THIS_SEED}" ]; then
  SEED=${THIS_SEED}
else
  SEED_LIST_ARRAY=($SEED_LIST)
  SEED=${SEED_LIST_ARRAY[$SLURM_ARRAY_TASK_ID]}
fi

if [ -z "${SEED}" ]; then
  echo "FATAL: Seed is empty. Check SEED_LIST and --array index." >&2
  exit 2
fi

INPUT_PREFIX=${INPUT_PREFIX:-circe_AA_dd_ps_nofilter_00_new}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-ddsim_C3_250_PS1_gg_had}
INPUT_FILE="${INPUT_PREFIX}_${SEED}.slcio"
OUTPUT_FILE="${OUTPUT_PREFIX}_${SEED}.edm4hep.root"

ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml --outputFile ${OUTPUT_FILE} --steeringFile ccc_steer_v2.py --inputFiles ${INPUT_FILE}  --vertexOffset 0 0 0 0 --numberOfEvents ${NUM_EVENTS:-5000}