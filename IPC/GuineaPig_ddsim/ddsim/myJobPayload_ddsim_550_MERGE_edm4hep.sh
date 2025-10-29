#!/bin/bash
# This is the payload script that runs the actual job commands

collider=$1
seed=$2

echo "Running with collider: $collider and seed: $seed"

cd C^3/bkg_studies_2023/GuineaPig_July_2024
#cd /fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

cd output_new/${collider}/ddsim


input_file="ddsim_${collider}_v2_seed_${seed}_MERGED.edm4hep.root"

output_file_A="ddsim_${collider}_v2_seed_${seed}_A.edm4hep.root"
output_file_B="ddsim_${collider}_v2_seed_${seed}_B.edm4hep.root"
output_file_C="ddsim_${collider}_v2_seed_${seed}_C.edm4hep.root"
output_file_D="ddsim_${collider}_v2_seed_${seed}_D.edm4hep.root"
output_file_E="ddsim_${collider}_v2_seed_${seed}_E.edm4hep.root"

hadd -f ${input_file} ${output_file_A} ${output_file_B} ${output_file_C} ${output_file_D} ${output_file_E}
