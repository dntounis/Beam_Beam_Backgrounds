cd /fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/whizard-3.1.5/
source runtime_env.sh

# Resolve array index to a seed value
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

cd ${WORK_DIR:-/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS1}

TEMPLATE=${SIN_TEMPLATE:-circe_AA_dd_gg_had_ps_C3_250_PS1.gg.sin}
BASENAME=${TEMPLATE%.gg.sin}
NEW_SIN=${BASENAME}_${SEED}.gg.sin

# Copy and update seed and output sample name inside the .sin file
cp -f "$TEMPLATE" "$NEW_SIN"

# Update the top-level seed and the seed inside iterations block if present.
# Be tolerant to optional spaces.
sed -i "s/^seed[[:space:]]*=[[:space:]]*.*/seed = ${SEED}/" "$NEW_SIN"
sed -i "s/^\([[:space:]]*seed[[:space:]]*=[[:space:]]*\).*/\1${SEED}/" "$NEW_SIN"

# Update the sample/output name to carry the seed suffix
# Append seed to $sample value. Works whether or not a seed suffix already exists.
sed -i "s/\(^\$sample = \"[^\"]*\)\(\"\)/\1_${SEED}\2/" "$NEW_SIN"

# Run whizard with the per-seed input
whizard "$NEW_SIN"
