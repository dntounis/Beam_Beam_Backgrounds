#!/bin/bash
# This is the payload script that runs the actual job commands

collider=$1
seed=$2
letter=$3 #A,B or C for 550GeV: splitting the input files in three

echo "Running with collider: $collider and seed: $seed and letter: $letter"

cd C^3/bkg_studies_2023/GuineaPig_July_2024
#cd /fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

# Jim: change output name to slcio or edm4hep.root accordingly!!!
ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
        --outputFile ddsim_${collider}_v2_seed_${seed}_${letter}.edm4hep.root \
        --steeringFile ccc_steer_v2.py \
        --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_${letter}.pairs \
        --vertexOffset 0 0 0 0 \
        --numberOfEvents 1 \
        --guineapig.particlesPerEvent -1

#mv ddsim_${collider}_seed_${seed}.edm4hep.root output_new/${collider}/ddsim/ddsim_${collider}_seed_${seed}.edm4hep.root

#mv ddsim_${collider}_seed_${seed}_${letter}.slcio output_new/${collider}/ddsim/ddsim_${collider}_seed_${seed}_${letter}.slcio
mv ddsim_${collider}_v2_seed_${seed}_${letter}.edm4hep.root output_new/${collider}/ddsim/ddsim_${collider}_v2_seed_${seed}_${letter}.edm4hep.root
