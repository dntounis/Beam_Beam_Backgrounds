#!/bin/bash
# This is the payload script that runs the actual job commands

collider=$1
seed=$2

echo "Running with collider: $collider and seed: $seed"

cd C^3/bkg_studies_2023/GuineaPig_July_2024
#cd /fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh


cp output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.dat  output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.pairs

# Jim: change output name to slcio or edm4hep.root accordingly!!!
ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
            --outputFile ddsim_${collider}_seed_${seed}.edm4hep.root \
            --steeringFile ccc_steer.py \
            --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.pairs \
            --vertexOffset 0 0 0 0 \
            --numberOfEvents 1 \
            --guineapig.particlesPerEvent -1


#mv ddsim_${collider}_seed_${seed}.slcio output_new/${collider}/ddsim/ddsim_${collider}_seed_${seed}.slcio
mv ddsim_${collider}_seed_${seed}.edm4hep.root output_new/${collider}/ddsim/ddsim_${collider}_seed_${seed}.edm4hep.root


# remove .pairs file to save space (original .dat file is preserved)
